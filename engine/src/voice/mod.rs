//! 100%-on-RTX-3080 Speech-to-Text (`Qwen3-ASR`), Text-to-Speech (`Qwen3-TTS` + `12Hz Code2Wav`),
//! and Native macOS `CoreAudio` Microphone/Speaker I/O.

pub mod asr;
pub mod audio;
mod code2wav;
pub mod tts;
