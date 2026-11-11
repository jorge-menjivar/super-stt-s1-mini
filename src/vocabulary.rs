// SPDX-License-Identifier: GPL-3.0-only
//! The user's vocabulary, kept through the rewrite.
//!
//! A backend that declares `[capabilities] context = true` is handed the
//! vocabulary of the dictation context in force, as an `x-stt-vocabulary`
//! header: a JSON array of terms. S1-mini has nowhere to read it. Its input is
//! the control line and the transcript, in the format it was trained on, and
//! its model card is explicit that anything else makes it hallucinate. So the
//! vocabulary is applied around the model instead:
//!
//! 1. Before the rewrite, each term in the transcript is written the way the
//!    vocabulary spells it, matched without regard to case, spaces or
//!    punctuation: `SuperVoice` becomes `Super Voice`.
//! 2. After it, the rewrite is lined up with the transcript word by word, and
//!    a term the transcript had and the rewrite lost is put back where the
//!    rewrite changed it. S1-mini comes from Superwhisper, and turned
//!    "SuperVoice" into "Superwhisper"; this is what undoes that.
//!
//! The context's other half, its prompt, is not read: S1-mini follows no
//! instruction but its control line.

use std::ops::Range;

/// The header the daemon sends the vocabulary in.
pub const HEADER: &str = "x-stt-vocabulary";

/// The shortest key a term is matched by. A one-letter term would match every
/// "a" and "I" in a transcript.
const SHORTEST_KEY: usize = 2;

/// How alike a span of the rewrite has to be to a term it lost to be taken
/// for a mangled copy of it: one minus their edit distance over the longer
/// one's length. "Superwhisper" is 0.58 of "supervoice".
const SIMILAR: f64 = 0.5;

/// The most cells the alignment fills: two thousand words against two
/// thousand. A longer transcript keeps its rewrite as it is.
const MOST_CELLS: usize = 4_000_000;

/// One term: as the user spells it, and the key it is matched by.
struct Term {
    spelling: String,
    key: String,
    /// The words in the spelling. A mangled copy is looked for across one
    /// more than this.
    words: usize,
}

/// The terms of the dictation context in force. Empty when there is none.
pub struct Vocabulary {
    /// Longest key first, so a term inside another ("Super" in "Super Voice")
    /// never takes the longer one's place.
    terms: Vec<Term>,
}

/// A word of a text, by its byte offsets. `core` is the word without the
/// punctuation around it or a possessive `'s`, which a replacement keeps.
struct Token {
    core: Range<usize>,
    key: String,
}

/// A term found in a text: its tokens, and which term it is.
struct Found {
    tokens: Range<usize>,
    term: usize,
}

impl Vocabulary {
    /// From the header's value. Absent or unreadable is an empty vocabulary.
    pub fn from_header(value: Option<&str>) -> Self {
        let Some(value) = value else {
            return Self::new(Vec::new());
        };
        match serde_json::from_str::<Vec<String>>(value) {
            Ok(terms) => Self::new(terms),
            Err(e) => {
                log::warn!("{HEADER} is not a JSON array of strings ({e}); ignoring it");
                Self::new(Vec::new())
            }
        }
    }

    fn new(terms: Vec<String>) -> Self {
        let mut terms: Vec<Term> = terms
            .into_iter()
            .filter_map(|term| {
                let spelling = term.trim().to_string();
                let key = key(&spelling);
                (key.chars().count() >= SHORTEST_KEY).then(|| Term {
                    words: spelling.split_whitespace().count(),
                    spelling,
                    key,
                })
            })
            .collect();
        terms.sort_by_key(|term| std::cmp::Reverse(term.key.len()));
        Self { terms }
    }

    /// `text` with each term written the way the vocabulary spells it.
    pub fn spell(&self, text: &str) -> String {
        if self.terms.is_empty() {
            return text.to_string();
        }
        let tokens = tokens(text);
        let edits = self
            .find(&tokens)
            .into_iter()
            .map(|found| (core(&tokens, &found.tokens), found.term))
            .collect();
        self.apply(text, edits)
    }

    /// `output` as [`spell`](Self::spell) writes it, with each term `input`
    /// had and the rewrite lost put back where the rewrite changed it.
    /// `input` is the transcript as `spell` returned it.
    pub fn restore(&self, input: &str, output: &str) -> String {
        if self.terms.is_empty() {
            return output.to_string();
        }
        let output = self.spell(output);
        let had = tokens(input);
        let has = tokens(&output);
        let found = self.find(&had);
        let kept = self.find(&has);

        // How many of each term the rewrite lost. A term it moved elsewhere
        // is not lost, and is not put back a second time.
        let mut lost = vec![0usize; self.terms.len()];
        for found in &found {
            lost[found.term] += 1;
        }
        for kept in &kept {
            lost[kept.term] = lost[kept.term].saturating_sub(1);
        }
        if lost.iter().all(|&n| n == 0) {
            return output;
        }
        let Some(matched) = align(&had, &has) else {
            return output;
        };

        let mut taken = vec![false; has.len()];
        for kept in &kept {
            taken[kept.tokens.clone()].fill(true);
        }
        let mut edits = Vec::new();
        for found in &found {
            if lost[found.term] == 0 {
                continue;
            }
            let term = &self.terms[found.term];
            if let Some(span) = changed_to(&has, &matched, &found.tokens, term, &taken) {
                log::debug!(
                    "putting back {:?}, which the rewrite changed",
                    term.spelling
                );
                taken[span.clone()].fill(true);
                edits.push((core(&has, &span), found.term));
                lost[found.term] -= 1;
            }
        }
        edits.sort_by_key(|(range, _)| range.start);
        self.apply(&output, edits)
    }

    /// Each term in `tokens`, left to right, never overlapping.
    fn find(&self, tokens: &[Token]) -> Vec<Found> {
        let mut found = Vec::new();
        let mut start = 0;
        while start < tokens.len() {
            let hit = self.terms.iter().enumerate().find_map(|(term, t)| {
                spanning(tokens, start, &t.key).map(|end| Found {
                    tokens: start..end,
                    term,
                })
            });
            match hit {
                Some(hit) => {
                    start = hit.tokens.end;
                    found.push(hit);
                }
                None => start += 1,
            }
        }
        found
    }

    /// `text` with each byte range written as its term. The ranges are in
    /// order and do not overlap.
    fn apply(&self, text: &str, edits: Vec<(Range<usize>, usize)>) -> String {
        let mut out = String::with_capacity(text.len());
        let mut at = 0;
        for (range, term) in edits {
            out.push_str(&text[at..range.start]);
            out.push_str(&self.terms[term].spelling);
            at = range.end;
        }
        out.push_str(&text[at..]);
        out
    }
}

/// The span of the rewrite a lost term was changed to, if one can be told.
///
/// The window is what the rewrite wrote between the nearest words on either
/// side of the term that it kept. When the term was all the transcript had
/// there, and the window is about as long, the window is what the term became,
/// even if it looks nothing like it: "it", say. Otherwise the term is looked
/// for in the window as a mangled copy, by spelling. A window that is empty
/// is a term the rewrite dropped, which is left dropped.
fn changed_to(
    has: &[Token],
    matched: &[Option<usize>],
    term_tokens: &Range<usize>,
    term: &Term,
    taken: &[bool],
) -> Option<Range<usize>> {
    if term_tokens.clone().all(|i| matched[i].is_some()) {
        return None;
    }
    let before = (0..term_tokens.start)
        .rev()
        .find_map(|i| matched[i].map(|j| (i, j)));
    let after = (term_tokens.end..matched.len()).find_map(|i| matched[i].map(|j| (i, j)));
    let from = before.map_or(0, |(_, j)| j + 1);
    let to = after.map_or(has.len(), |(_, j)| j);
    if from >= to {
        return None;
    }

    let only_the_term = before.map_or(0, |(i, _)| i + 1) == term_tokens.start
        && after.map_or(matched.len(), |(i, _)| i) == term_tokens.end;
    if only_the_term && to - from <= term.words + 1 && !taken[from..to].contains(&true) {
        return Some(from..to);
    }

    let mut best: Option<(f64, Range<usize>)> = None;
    for start in from..to {
        let mut joined = String::new();
        for end in start..(start + term.words + 1).min(to) {
            if taken[end] {
                break;
            }
            joined.push_str(&has[end].key);
            let score = similarity(&joined, &term.key);
            if score >= SIMILAR && best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, start..end + 1));
            }
        }
    }
    best.map(|(_, span)| span)
}

/// For each token of `had`, the token of `has` it lines up with, if any: a
/// longest common sequence of their keys. `None` when the two are too long to
/// line up.
fn align(had: &[Token], has: &[Token]) -> Option<Vec<Option<usize>>> {
    let (n, m) = (had.len(), has.len());
    if n.saturating_mul(m) > MOST_CELLS {
        return None;
    }
    // `run[i * width + j]`: the longest common sequence of `had[i..]` and
    // `has[j..]`.
    let width = m + 1;
    let mut run = vec![0u32; (n + 1) * width];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            run[i * width + j] = if had[i].key == has[j].key {
                run[(i + 1) * width + j + 1] + 1
            } else {
                run[(i + 1) * width + j].max(run[i * width + j + 1])
            };
        }
    }
    let mut matched = vec![None; n];
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if had[i].key == has[j].key {
            matched[i] = Some(j);
            i += 1;
            j += 1;
        } else if run[(i + 1) * width + j] >= run[i * width + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(matched)
}

/// The end of the span from `start` whose keys, run together, are `key`.
fn spanning(tokens: &[Token], start: usize, key: &str) -> Option<usize> {
    let mut joined = String::new();
    for (end, token) in tokens.iter().enumerate().skip(start) {
        if token.key.is_empty() {
            return None;
        }
        joined.push_str(&token.key);
        if joined == key {
            return Some(end + 1);
        }
        if !key.starts_with(joined.as_str()) {
            return None;
        }
    }
    None
}

/// The byte range from the first token's core to the last one's.
fn core(tokens: &[Token], span: &Range<usize>) -> Range<usize> {
    tokens[span.start].core.start..tokens[span.end - 1].core.end
}

/// The words of `text`, split at whitespace.
fn tokens(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (at, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        match (start, c.is_whitespace()) {
            (None, false) => start = Some(at),
            (Some(from), true) => {
                tokens.push(token(text, from..at));
                start = None;
            }
            _ => {}
        }
    }
    tokens
}

/// One word: its core is the run from its first letter or digit to its last,
/// less a possessive `'s`.
fn token(text: &str, word: Range<usize>) -> Token {
    let w = &text[word.clone()];
    let Some(first) = w.find(char::is_alphanumeric) else {
        return Token {
            core: word.end..word.end,
            key: String::new(),
        };
    };
    let last = w
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(w.len(), |(i, c)| i + c.len_utf8());
    let mut end = last;
    for suffix in ["'s", "’s"] {
        if w[first..last].len() > suffix.len() && w[first..last].ends_with(suffix) {
            end = last - suffix.len();
            break;
        }
    }
    let core = word.start + first..word.start + end;
    Token {
        key: key(&text[core.clone()]),
        core,
    }
}

/// What a term or word is matched by: its letters and digits, lowercase.
fn key(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// One minus the edit distance of `a` and `b` over the longer one's length.
fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let longer = a.len().max(b.len());
    if longer == 0 {
        return 1.0;
    }
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if ca == cb {
                diagonal
            } else {
                1 + diagonal.min(above).min(row[j])
            };
            diagonal = above;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let distance = row[b.len()] as f64 / longer as f64;
    1.0 - distance
}

#[cfg(test)]
mod tests {
    use super::{HEADER, Vocabulary, similarity};

    fn vocabulary(terms: &[&str]) -> Vocabulary {
        Vocabulary::new(terms.iter().map(ToString::to_string).collect())
    }

    #[test]
    fn spells_a_term_the_way_the_vocabulary_does() {
        let v = vocabulary(&["Super Voice"]);
        assert_eq!(
            v.spell("I updated SuperVoice today"),
            "I updated Super Voice today"
        );
        assert_eq!(v.spell("about super voice, and"), "about Super Voice, and");
        assert_eq!(v.spell("SuperVoice's daemon"), "Super Voice's daemon");
    }

    #[test]
    fn spells_only_whole_words() {
        let v = vocabulary(&["Super Voice"]);
        assert_eq!(v.spell("a SuperVoiceApp build"), "a SuperVoiceApp build");
        assert_eq!(v.spell("super cool voice"), "super cool voice");
    }

    #[test]
    fn the_longer_term_wins() {
        let v = vocabulary(&["Super", "Super Voice"]);
        assert_eq!(v.spell("supervoice and super"), "Super Voice and Super");
    }

    #[test]
    fn a_one_letter_term_is_ignored() {
        let v = vocabulary(&["A", "STT"]);
        assert_eq!(v.spell("a stt b"), "a STT b");
    }

    /// What it was written for: the rewrite turned the product's name into
    /// Superwhisper's, and changed the rest as it should.
    #[test]
    fn puts_back_a_term_the_rewrite_changed() {
        let v = vocabulary(&["Super Voice"]);
        let input = v.spell(
            "I have updated SuperVoice to the latest version, which is the version beta three.",
        );
        let output = "I have updated Superwhisper to the latest version, which is version beta 3.";
        assert_eq!(
            v.restore(&input, output),
            "I have updated Super Voice to the latest version, which is version beta 3."
        );
    }

    /// Where the rewrite changed the words around the term too, the term goes
    /// back on the word that looks like it.
    #[test]
    fn puts_back_a_term_among_other_changes() {
        let v = vocabulary(&["Super Voice"]);
        let input = v.spell("we use super voice daily");
        assert_eq!(
            v.restore(&input, "We use Superwhisper every day."),
            "We use Super Voice every day."
        );
    }

    /// Alone between kept words, the term is put back over whatever the
    /// rewrite wrote in its place, however unlike it.
    #[test]
    fn puts_back_a_term_the_rewrite_replaced_outright() {
        let v = vocabulary(&["STT"]);
        let input = v.spell("the stt daemon is up");
        assert_eq!(
            v.restore(&input, "The speech-to-text daemon is up."),
            "The STT daemon is up."
        );
    }

    #[test]
    fn keeps_a_rewrite_that_kept_the_term() {
        let v = vocabulary(&["Super Voice"]);
        let input = v.spell("um so super voice is uh great");
        assert_eq!(
            v.restore(&input, "So supervoice is great."),
            "So Super Voice is great."
        );
    }

    #[test]
    fn a_term_the_rewrite_moved_is_not_put_back_twice() {
        let v = vocabulary(&["Super Voice"]);
        let input = v.spell("super voice is what I use");
        assert_eq!(
            v.restore(&input, "What I use is Super Voice."),
            "What I use is Super Voice."
        );
    }

    #[test]
    fn a_term_the_rewrite_dropped_stays_dropped() {
        let v = vocabulary(&["Gravvit"]);
        let input = v.spell("so gravvit uh yes");
        assert_eq!(v.restore(&input, "So, yes."), "So, yes.");
    }

    /// Words the rewrite wrote in a term's place, more than the term had and
    /// none like it, are not taken for it.
    #[test]
    fn leaves_unlike_words_alone() {
        let v = vocabulary(&["Gravvit"]);
        let input = v.spell("we shipped gravvit and then lunch");
        assert_eq!(
            v.restore(&input, "We shipped the new mobile app and then lunch."),
            "We shipped the new mobile app and then lunch."
        );
    }

    #[test]
    fn no_vocabulary_changes_nothing() {
        let v = vocabulary(&[]);
        assert_eq!(v.spell("SuperVoice"), "SuperVoice");
        assert_eq!(v.restore("SuperVoice", "Superwhisper"), "Superwhisper");
    }

    #[test]
    fn reads_the_header() {
        let v = Vocabulary::from_header(Some(r#"["Super Voice","Gravvit"]"#));
        assert_eq!(v.spell("supervoice gravvit"), "Super Voice Gravvit");
        assert!(Vocabulary::from_header(None).terms.is_empty());
        assert!(
            Vocabulary::from_header(Some("Super Voice"))
                .terms
                .is_empty(),
            "{HEADER} is JSON; anything else is ignored"
        );
    }

    /// The daemon escapes the header to ASCII; the JSON parser undoes it.
    #[test]
    fn reads_a_term_escaped_to_ascii() {
        let v = Vocabulary::from_header(Some(r#"["Menjívar"]"#));
        assert_eq!(v.spell("menjívar"), "Menjívar");
    }

    #[test]
    fn similarity_is_one_minus_the_edit_distance_over_the_length() {
        assert!((similarity("supervoice", "superwhisper") - 7.0 / 12.0).abs() < 1e-9);
        assert!((similarity("abc", "abc") - 1.0).abs() < 1e-9);
    }
}
