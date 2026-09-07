// SPDX-License-Identifier: GPL-3.0-only
//! Splitting a long transcript into pieces the model can take.
//!
//! S1-mini is trained on inputs of up to about a thousand tokens; past that
//! the output degrades. A dictated transcript can run longer, so one that
//! overflows the budget is cut into pieces at sentence boundaries — falling
//! back to word boundaries for a sentence that is itself too long — and each
//! piece is normalized on its own. The model card recommends exactly this.
//!
//! Token counting is injected so the chunker is pure: the engine passes the
//! real tokenizer, the tests pass a word counter.

/// Split `text` into pieces of at most `budget` tokens as measured by `count`.
///
/// Text within the budget comes back as a single piece, untouched. Otherwise
/// sentences are packed greedily in order, and any single sentence over the
/// budget is packed word by word. The pieces concatenate back to the original
/// words in the original order; only whitespace between pieces is dropped.
pub fn chunk(text: &str, budget: usize, count: impl Fn(&str) -> usize) -> Vec<String> {
    let budget = budget.max(1);
    if count(text) <= budget {
        return vec![text.to_string()];
    }

    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0;

    for sentence in sentences(text) {
        let sentence_tokens = count(sentence);
        if sentence_tokens > budget {
            // Too long to place whole. Flush what is pending, then pack the
            // sentence's words the same way — a sentence is just a smaller
            // transcript with no boundaries left to prefer.
            flush(&mut pieces, &mut current, &mut current_tokens);
            for word in sentence.split_whitespace() {
                push(
                    &mut pieces,
                    &mut current,
                    &mut current_tokens,
                    word,
                    count(word),
                    budget,
                );
            }
            continue;
        }
        push(
            &mut pieces,
            &mut current,
            &mut current_tokens,
            sentence,
            sentence_tokens,
            budget,
        );
    }
    flush(&mut pieces, &mut current, &mut current_tokens);
    pieces
}

/// Append `unit` to the pending piece, flushing first if it would overflow.
fn push(
    pieces: &mut Vec<String>,
    current: &mut String,
    current_tokens: &mut usize,
    unit: &str,
    unit_tokens: usize,
    budget: usize,
) {
    if !current.is_empty() && *current_tokens + unit_tokens > budget {
        flush(pieces, current, current_tokens);
    }
    if !current.is_empty() {
        current.push(' ');
    }
    current.push_str(unit);
    *current_tokens += unit_tokens;
}

fn flush(pieces: &mut Vec<String>, current: &mut String, current_tokens: &mut usize) {
    if !current.is_empty() {
        pieces.push(std::mem::take(current));
    }
    *current_tokens = 0;
}

/// Split at sentence boundaries: a `.`, `?` or `!` followed by whitespace, or a
/// newline. Each returned slice is trimmed and non-empty.
///
/// A raw transcript is usually unpunctuated — that is what the model is for —
/// so this often yields one "sentence", and the word-level fallback in
/// [`chunk`] does the work. When punctuation is present it is the better cut.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary = match c {
            '\n' => true,
            '.' | '?' | '!' => chars.peek().is_none_or(|(_, next)| next.is_whitespace()),
            _ => false,
        };
        if boundary {
            let end = i + c.len_utf8();
            let s = text[start..end].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// Join the normalized pieces back into one transcript.
///
/// Pieces are prose and join with a space, unless either side carries a line
/// break — a bullet list or an email layout — in which case a newline keeps the
/// layout the model produced. Empty pieces (a chunk that was all filler) are
/// dropped rather than leaving a stray separator.
#[must_use]
pub fn join(pieces: &[String]) -> String {
    let mut out = String::new();
    for piece in pieces.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        if !out.is_empty() {
            out.push(if out.contains('\n') || piece.contains('\n') {
                '\n'
            } else {
                ' '
            });
        }
        out.push_str(piece);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{chunk, join, sentences};

    fn words(s: &str) -> usize {
        s.split_whitespace().count()
    }

    #[test]
    fn text_within_budget_is_one_untouched_piece() {
        let text = "so um send the report   by friday";
        assert_eq!(chunk(text, 10, words), vec![text.to_string()]);
    }

    #[test]
    fn sentences_split_at_terminal_punctuation_and_newlines() {
        assert_eq!(
            sentences("Send it. Did you? Yes!\nThanks a lot"),
            vec!["Send it.", "Did you?", "Yes!", "Thanks a lot"]
        );
    }

    /// A period only ends a sentence when whitespace follows, so decimals,
    /// abbreviations and addresses stay whole.
    #[test]
    fn interior_periods_are_not_boundaries() {
        assert_eq!(
            sentences("the total is 3.5 at support@example.com today. ok"),
            vec!["the total is 3.5 at support@example.com today.", "ok"]
        );
    }

    #[test]
    fn sentences_pack_greedily_under_the_budget() {
        let text = "one two three. four five. six seven eight. nine.";
        assert_eq!(
            chunk(text, 5, words),
            vec![
                "one two three. four five.".to_string(),
                "six seven eight. nine.".to_string()
            ]
        );
    }

    /// An unpunctuated transcript — the common case — has no sentence to cut
    /// at, so it is cut at words instead, and no word is ever split.
    #[test]
    fn an_unpunctuated_run_falls_back_to_word_boundaries() {
        let text = "a b c d e f g h i j k";
        assert_eq!(
            chunk(text, 4, words),
            vec![
                "a b c d".to_string(),
                "e f g h".to_string(),
                "i j k".to_string()
            ]
        );
    }

    /// One over-long sentence in the middle is packed by words without
    /// disturbing the sentences around it.
    #[test]
    fn an_oversized_sentence_is_split_without_merging_into_its_neighbours() {
        let text = "short one. a b c d e f g. last one.";
        assert_eq!(
            chunk(text, 4, words),
            vec![
                "short one.".to_string(),
                "a b c d".to_string(),
                "e f g.".to_string(),
                "last one.".to_string()
            ]
        );
    }

    /// Chunking loses nothing but inter-piece whitespace: the words come back
    /// in order, every one of them.
    #[test]
    fn chunking_preserves_every_word_in_order() {
        let text = "the quick brown fox. jumps over the lazy dog! and then? it slept\nfor a while";
        for budget in 1..12 {
            let pieces = chunk(text, budget, words);
            let rejoined: Vec<&str> = pieces.iter().flat_map(|p| p.split_whitespace()).collect();
            let original: Vec<&str> = text.split_whitespace().collect();
            assert_eq!(rejoined, original, "budget {budget}");
            for piece in &pieces {
                assert!(words(piece) <= budget, "budget {budget}: {piece:?}");
            }
        }
    }

    #[test]
    fn a_zero_budget_still_terminates_one_word_per_piece() {
        assert_eq!(
            chunk("a b c", 0, words),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn prose_pieces_join_with_a_space() {
        assert_eq!(
            join(&["First part.".into(), "Second part.".into()]),
            "First part. Second part."
        );
    }

    /// A list or an email layout carries newlines the model chose; the next
    /// piece must not land on the same line as the last bullet.
    #[test]
    fn pieces_with_line_breaks_join_on_a_new_line() {
        assert_eq!(
            join(&["We need:\n- A\n- B".into(), "Then we leave.".into()]),
            "We need:\n- A\n- B\nThen we leave."
        );
    }

    /// A chunk that was all filler normalizes to nothing, and that nothing
    /// must not become a doubled separator.
    #[test]
    fn empty_pieces_are_dropped() {
        assert_eq!(
            join(&["Hello.".into(), String::new(), "  ".into(), "Bye.".into()]),
            "Hello. Bye."
        );
        assert_eq!(join(&[String::new()]), "");
    }
}
