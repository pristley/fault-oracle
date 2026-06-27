//! Symbol alphabet management
//!
//! Defines alphabets for symbolic encoding.
//! Implements Phase 4.3: Alphabet Management

use serde::{Deserialize, Serialize};
use crate::Result;

/// Symbol alphabet for encoding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Alphabet {
    /// Binary alphabet {0, 1}
    Binary,
    /// Quaternary alphabet {0, 1, 2, 3}
    Quaternary,
    /// Generic alphabet with n symbols (a, b, c, ...)
    Generic(usize),
    /// Custom symbols
    Custom(Vec<char>),
}

impl Alphabet {
    /// Create an alphabet with the specified size
    ///
    /// Generates symbols a, b, c, ... up to the specified size.
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 || size > 26 {
            return Err(crate::SdfError::InvalidParameter(
                "Alphabet size must be between 1 and 26".to_string(),
            ));
        }

        Ok(Alphabet::Generic(size))
    }

    /// Create a binary alphabet {a, b}
    pub fn binary() -> Self {
        Alphabet::Binary
    }

    /// Create a quaternary alphabet {a, b, c, d}
    pub fn quaternary() -> Self {
        Alphabet::Quaternary
    }

    /// Create an alphabet from explicit symbols
    pub fn from_symbols(symbols: Vec<char>) -> Result<Self> {
        if symbols.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Symbols cannot be empty".to_string(),
            ));
        }

        if symbols.len() > 26 {
            return Err(crate::SdfError::InvalidParameter(
                "Cannot have more than 26 symbols".to_string(),
            ));
        }

        Ok(Alphabet::Custom(symbols))
    }

    /// Get a symbol by index
    ///
    /// Implements Phase 4.3, Step 1: nth_symbol
    pub fn symbol(&self, index: usize) -> Result<char> {
        let symbols = self.symbols();
        if index >= symbols.len() {
            return Err(crate::SdfError::InvalidParameter(
                format!("Symbol index {} out of bounds", index),
            ));
        }
        Ok(symbols[index])
    }

    /// Get the size of the alphabet
    ///
    /// Implements Phase 4.3, Step 1: size
    pub fn size(&self) -> usize {
        self.symbols().len()
    }

    /// Get all symbols
    ///
    /// Implements Phase 4.3, Step 2: symbols
    pub fn symbols(&self) -> Vec<char> {
        match self {
            Alphabet::Binary => vec!['a', 'b'],
            Alphabet::Quaternary => vec!['a', 'b', 'c', 'd'],
            Alphabet::Generic(n) => (0..*n)
                .map(|i| (('a' as u8 + i as u8) as char))
                .collect(),
            Alphabet::Custom(symbols) => symbols.clone(),
        }
    }

    /// Get symbol at position (safe wrapper)
    ///
    /// Implements Phase 4.3, Step 3: nth_symbol
    pub fn nth_symbol(&self, n: usize) -> Option<char> {
        self.symbol(n).ok()
    }

    /// Check if a character is in this alphabet
    pub fn contains(&self, c: char) -> bool {
        self.symbols().contains(&c)
    }

    /// Get the index of a symbol
    pub fn index_of(&self, c: char) -> Option<usize> {
        self.symbols().iter().position(|&sym| sym == c)
    }
}

impl Default for Alphabet {
    fn default() -> Self {
        Alphabet::Generic(3)
    }
}

impl std::fmt::Display for Alphabet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Alphabet::Binary => write!(f, "Binary{{a,b}}"),
            Alphabet::Quaternary => write!(f, "Quaternary{{a,b,c,d}}"),
            Alphabet::Generic(n) => {
                let symbols: String = self.symbols().iter().collect();
                write!(f, "Alphabet(n={}; {{{}}})", n, symbols)
            }
            Alphabet::Custom(symbols) => {
                let symbols_str: String = symbols.iter().collect();
                write!(f, "Custom{{{}}}",symbols_str)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alphabet_creation() {
        let alphabet = Alphabet::new(5).unwrap();
        assert_eq!(alphabet.size(), 5);
        assert_eq!(alphabet.symbol(0).unwrap(), 'a');
        assert_eq!(alphabet.symbol(4).unwrap(), 'e');
    }

    #[test]
    fn test_alphabet_binary() {
        let alphabet = Alphabet::binary();
        assert_eq!(alphabet.size(), 2);
        assert_eq!(alphabet.symbols(), vec!['a', 'b']);
    }

    #[test]
    fn test_alphabet_quaternary() {
        let alphabet = Alphabet::quaternary();
        assert_eq!(alphabet.size(), 4);
        assert_eq!(alphabet.symbols(), vec!['a', 'b', 'c', 'd']);
    }

    #[test]
    fn test_alphabet_custom() {
        let symbols = vec!['x', 'y', 'z'];
        let alphabet = Alphabet::from_symbols(symbols).unwrap();
        assert_eq!(alphabet.size(), 3);
        assert_eq!(alphabet.nth_symbol(1), Some('y'));
    }

    #[test]
    fn test_alphabet_invalid_size() {
        assert!(Alphabet::new(0).is_err());
        assert!(Alphabet::new(27).is_err());
    }

    #[test]
    fn test_alphabet_contains() {
        let alphabet = Alphabet::new(3).unwrap();
        assert!(alphabet.contains('a'));
        assert!(alphabet.contains('b'));
        assert!(!alphabet.contains('z'));
    }

    #[test]
    fn test_alphabet_index_of() {
        let alphabet = Alphabet::new(4).unwrap();
        assert_eq!(alphabet.index_of('a'), Some(0));
        assert_eq!(alphabet.index_of('c'), Some(2));
        assert_eq!(alphabet.index_of('z'), None);
    }
}
