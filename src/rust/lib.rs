//! Minimal library crate to anchor workspace tests.

#[cfg(test)]
mod tests {
    #[test]
    fn workspace_anchor() {
        assert_eq!(2 + 2, 4);
    }
}

