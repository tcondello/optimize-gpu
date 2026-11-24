//! Multi-Line Text Composer with Cursor Navigation, Word Kill, History Recall, and Visual Line Wrapping.

use super::terminal::char_display_width;

pub(super) struct Composer {
    chars: Vec<char>,
    cursor_idx: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
}

impl Composer {
    pub(super) fn new() -> Self {
        Self {
            chars: Vec::new(),
            cursor_idx: 0,
            history: Vec::new(),
            history_pos: None,
        }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub(super) fn clear(&mut self) {
        self.chars.clear();
        self.cursor_idx = 0;
        self.history_pos = None;
    }

    pub(super) fn insert_char(&mut self, ch: char) {
        self.chars.insert(self.cursor_idx, ch);
        self.cursor_idx += 1;
    }

    pub(super) fn insert_str(&mut self, text: &str) {
        for ch in text.chars() {
            self.chars.insert(self.cursor_idx, ch);
            self.cursor_idx += 1;
        }
    }

    pub(super) fn backspace(&mut self) {
        if self.cursor_idx > 0 {
            self.cursor_idx -= 1;
            self.chars.remove(self.cursor_idx);
        }
    }

    pub(super) fn delete_forward(&mut self) {
        if self.cursor_idx < self.chars.len() {
            self.chars.remove(self.cursor_idx);
        }
    }

    pub(super) fn delete_word_back(&mut self) {
        while self.cursor_idx > 0 && self.chars[self.cursor_idx - 1] == ' ' {
            self.cursor_idx -= 1;
            self.chars.remove(self.cursor_idx);
        }
        while self.cursor_idx > 0
            && self.chars[self.cursor_idx - 1] != ' '
            && self.chars[self.cursor_idx - 1] != '\n'
        {
            self.cursor_idx -= 1;
            self.chars.remove(self.cursor_idx);
        }
    }

    pub(super) fn kill_to_start(&mut self) {
        self.chars.drain(..self.cursor_idx);
        self.cursor_idx = 0;
    }

    pub(super) fn move_left(&mut self) {
        self.cursor_idx = self.cursor_idx.saturating_sub(1);
    }

    pub(super) fn move_right(&mut self) {
        if self.cursor_idx < self.chars.len() {
            self.cursor_idx += 1;
        }
    }

    pub(super) fn move_home(&mut self) {
        while self.cursor_idx > 0 && self.chars[self.cursor_idx - 1] != '\n' {
            self.cursor_idx -= 1;
        }
    }

    pub(super) fn move_end(&mut self) {
        while self.cursor_idx < self.chars.len() && self.chars[self.cursor_idx] != '\n' {
            self.cursor_idx += 1;
        }
    }

    pub(super) fn move_vertical(&mut self, dir: isize) {
        let before: String = self.chars[..self.cursor_idx].iter().collect();
        let after: String = self.chars[self.cursor_idx..].iter().collect();

        if dir < 0 && !before.contains('\n') {
            if self.history.is_empty() {
                return;
            }
            let next_pos = match self.history_pos {
                None => self.history.len() - 1,
                Some(p) => p.saturating_sub(1),
            };
            self.history_pos = Some(next_pos);
            self.chars = self.history[next_pos].chars().collect();
            self.cursor_idx = self.chars.len();
        } else if dir > 0 && !after.contains('\n') {
            if let Some(p) = self.history_pos {
                if p + 1 < self.history.len() {
                    self.history_pos = Some(p + 1);
                    self.chars = self.history[p + 1].chars().collect();
                    self.cursor_idx = self.chars.len();
                } else {
                    self.history_pos = None;
                    self.chars.clear();
                    self.cursor_idx = 0;
                }
            }
        } else if dir < 0 {
            if let Some(prev_nl) = before.rfind('\n') {
                let col_chars = before[prev_nl + 1..].chars().count();
                let prev_line_start = before[..prev_nl].rfind('\n').map(|p| p + 1).unwrap_or(0);
                let prev_line_chars = before[prev_line_start..prev_nl].chars().count();
                let prev_start_cidx = before[..prev_line_start].chars().count();
                self.cursor_idx = prev_start_cidx + col_chars.min(prev_line_chars);
            }
        } else if let Some(next_nl_rel) = after.find('\n') {
            let cur_line_start = before.rfind('\n').map(|p| p + 1).unwrap_or(0);
            let col_chars = before[cur_line_start..].chars().count();
            let rest_after_nl = &after[next_nl_rel + 1..];
            let next_line_len = rest_after_nl
                .find('\n')
                .map(|p| rest_after_nl[..p].chars().count())
                .unwrap_or_else(|| rest_after_nl.chars().count());
            let next_line_start_cidx = self.cursor_idx + after[..next_nl_rel].chars().count() + 1;
            self.cursor_idx = next_line_start_cidx + col_chars.min(next_line_len);
        }
    }

    pub(super) fn take_submission(&mut self) -> Option<String> {
        let submitted: String = self.chars.iter().collect();
        let trimmed = submitted.trim().to_string();
        if trimmed.is_empty() {
            return None;
        }
        self.chars.clear();
        self.cursor_idx = 0;
        self.history_pos = None;
        if self.history.last().map(|s| s.as_str()) != Some(trimmed.as_str()) {
            self.history.push(trimmed.clone());
        }
        Some(trimmed)
    }

    pub(super) fn wrapped_lines(&self, inner_w: usize) -> (Vec<String>, usize, usize) {
        let w = inner_w.max(8);
        let mut lines = vec![String::new()];
        let mut col = 0usize;
        let mut cur_row = 0usize;
        let mut cur_col = 0usize;

        for (idx, &ch) in self.chars.iter().enumerate() {
            if idx == self.cursor_idx {
                cur_row = lines.len() - 1;
                cur_col = col;
            }
            if ch == '\n' {
                lines.push(String::new());
                col = 0;
            } else {
                let cw = char_display_width(ch).max(1);
                if col + cw > w {
                    lines.push(String::new());
                    col = 0;
                }
                lines.last_mut().unwrap().push(ch);
                col += cw;
            }
        }
        if self.cursor_idx >= self.chars.len() {
            cur_row = lines.len() - 1;
            cur_col = col;
        }
        (lines, cur_row, cur_col)
    }
}
