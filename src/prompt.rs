// SPDX-License-Identifier: GPL-3.0-only
//! The prompt S1-mini was trained on, and the control line that steers it.
//!
//! Everything here is a literal from the model card
//! (<https://huggingface.co/superwhisper/s1-mini>). The card is explicit that
//! the wording is part of the input format: change the system prompt, drop the
//! control line, or send a value outside the trained sets, and the model
//! hallucinates or emits nothing. So the strings are constants, the axes are
//! closed enums, and an option value nobody can parse falls back to its
//! default rather than reaching the model.

use std::fmt;

/// The exact system prompt. Do not reword it.
pub const SYSTEM_PROMPT: &str = "You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.";

/// The assistant turn opens with an *empty* think block: S1-mini was trained
/// with thinking off, and without this prefix it emits an empty `<think>` and
/// stops. Written out: `<|im_start|>assistant\n<think>\n\n</think>\n\n` — two
/// newlines inside the block, two more after it.
pub const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The register: how much of the speaker's voice survives into writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Styling {
    /// Everything lowercase, apostrophes stripped, colloquialisms kept.
    Casual,
    /// The speaker's phrasing, with `I` and its contractions capitalized.
    SemiCasual,
    /// Standard written English; contractions kept, colloquialisms smoothed.
    #[default]
    SemiFormal,
    /// `semi-formal` with contractions expanded (`I am`, `cannot`).
    Formal,
}

/// Whether enumerable content may become a Markdown bullet list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Structure {
    #[default]
    Prose,
    Lists,
}

/// Destination conventions. `email` lays out a greeting, body and sign-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Context {
    #[default]
    General,
    Email,
}

/// One trained value of a control-line axis: its wire spelling, and how to
/// read it back from an option header.
pub trait Axis: Copy + Default + fmt::Debug {
    /// The option name the settings UI stores the value under.
    const OPTION: &'static str;

    /// The value as it appears on the control line.
    fn as_str(self) -> &'static str;

    /// Parse a trained value. `None` for anything else — the caller decides
    /// what an unknown value falls back to.
    fn parse(value: &str) -> Option<Self>;
}

impl Axis for Styling {
    const OPTION: &'static str = "styling";

    fn as_str(self) -> &'static str {
        match self {
            Self::Casual => "casual",
            Self::SemiCasual => "semi-casual",
            Self::SemiFormal => "semi-formal",
            Self::Formal => "formal",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "casual" => Some(Self::Casual),
            "semi-casual" => Some(Self::SemiCasual),
            "semi-formal" => Some(Self::SemiFormal),
            "formal" => Some(Self::Formal),
            _ => None,
        }
    }
}

impl Axis for Structure {
    const OPTION: &'static str = "structure";

    fn as_str(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Lists => "lists",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "prose" => Some(Self::Prose),
            "lists" => Some(Self::Lists),
            _ => None,
        }
    }
}

impl Axis for Context {
    const OPTION: &'static str = "context";

    fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Email => "email",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "general" => Some(Self::General),
            "email" => Some(Self::Email),
            _ => None,
        }
    }
}

/// The three axes of one request. Independent; every combination was trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Control {
    pub styling: Styling,
    pub structure: Structure,
    pub context: Context,
}

impl Control {
    /// Build the control for a request from its `[[options]]` values.
    ///
    /// `lookup` resolves an option name to the value the user set, if any.
    /// Values are trimmed and lowercased first, so `Semi-Formal ` reads as the
    /// trained `semi-formal`. A value that still is not one of the trained
    /// spellings is logged and replaced by the axis default: the model card
    /// warns that an untrained value garbles the output, and a typo in a
    /// settings field must not do that.
    pub fn from_options(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            styling: read_axis(&lookup),
            structure: read_axis(&lookup),
            context: read_axis(&lookup),
        }
    }

    /// The control line, exactly as the model expects it.
    #[must_use]
    pub fn line(self) -> String {
        format!(
            "[Styling: {}] [Structure: {}] [Context: {}]",
            self.styling.as_str(),
            self.structure.as_str(),
            self.context.as_str()
        )
    }
}

fn read_axis<A: Axis>(lookup: &impl Fn(&str) -> Option<String>) -> A {
    let Some(raw) = lookup(A::OPTION) else {
        return A::default();
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return A::default();
    }
    A::parse(&normalized).unwrap_or_else(|| {
        log::warn!(
            "option {}={raw:?} is not a value S1-mini was trained on; using {:?}",
            A::OPTION,
            A::default()
        );
        A::default()
    })
}

/// The full prompt for one transcript: system turn, user turn (control line,
/// newline, transcript), and the assistant prefix with thinking disabled.
#[must_use]
pub fn format_prompt(transcript: &str, control: Control) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{}\n{transcript}<|im_end|>\n\
         {ASSISTANT_PREFIX}",
        control.line()
    )
}

#[cfg(test)]
mod tests {
    use super::{ASSISTANT_PREFIX, Axis, Context, Control, Structure, Styling, format_prompt};

    #[test]
    fn the_defaults_are_the_model_cards_recommended_settings() {
        assert_eq!(
            Control::default().line(),
            "[Styling: semi-formal] [Structure: prose] [Context: general]"
        );
    }

    /// Every trained spelling round-trips through parse and back, so the
    /// control line can only ever carry a value the model has seen.
    #[test]
    fn every_trained_value_round_trips() {
        for s in [
            Styling::Casual,
            Styling::SemiCasual,
            Styling::SemiFormal,
            Styling::Formal,
        ] {
            assert_eq!(Styling::parse(s.as_str()), Some(s));
        }
        for s in [Structure::Prose, Structure::Lists] {
            assert_eq!(Structure::parse(s.as_str()), Some(s));
        }
        for c in [Context::General, Context::Email] {
            assert_eq!(Context::parse(c.as_str()), Some(c));
        }
    }

    #[test]
    fn options_set_exactly_their_own_axis() {
        let control = Control::from_options(|name| match name {
            "styling" => Some("formal".into()),
            "context" => Some("email".into()),
            _ => None,
        });
        assert_eq!(
            control,
            Control {
                styling: Styling::Formal,
                structure: Structure::Prose,
                context: Context::Email,
            }
        );
    }

    /// The settings UI stores whatever the user typed; case and padding are
    /// not what the model card is warning about, so they are forgiven.
    #[test]
    fn option_values_are_trimmed_and_case_folded() {
        let control = Control::from_options(|name| match name {
            "styling" => Some("  Semi-Casual\n".into()),
            "structure" => Some("LISTS".into()),
            _ => None,
        });
        assert_eq!(control.styling, Styling::SemiCasual);
        assert_eq!(control.structure, Structure::Lists);
    }

    /// An untrained value must never reach the model: it falls back to the
    /// axis default instead of garbling the output.
    #[test]
    fn an_unknown_or_blank_value_falls_back_to_the_default() {
        let control = Control::from_options(|name| match name {
            "styling" => Some("shouty".into()),
            "structure" => Some("   ".into()),
            "context" => Some("slack".into()),
            _ => None,
        });
        assert_eq!(control, Control::default());
    }

    #[test]
    fn the_prompt_is_the_literal_from_the_model_card() {
        let prompt = format_prompt("so um send the report by uh friday", Control::default());
        assert_eq!(
            prompt,
            "<|im_start|>system\n\
             You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>\n\
             <|im_start|>user\n\
             [Styling: semi-formal] [Structure: prose] [Context: general]\n\
             so um send the report by uh friday<|im_end|>\n\
             <|im_start|>assistant\n\
             <think>\n\
             \n\
             </think>\n\
             \n"
        );
    }

    /// The single most common way to get a blank answer from this model is
    /// a missing or malformed think block, so its exact shape is pinned.
    #[test]
    fn the_assistant_prefix_carries_an_empty_think_block() {
        assert!(ASSISTANT_PREFIX.ends_with("<think>\n\n</think>\n\n"));
    }
}
