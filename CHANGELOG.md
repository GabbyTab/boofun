# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core BooleanFunction class with multiple representation support
- Truth table representation with comprehensive evaluation methods
- Built-in Boolean functions: majority, dictator, tribes, parity, constant
- SpectralAnalyzer for Fourier analysis and influence computation
- Basic PropertyTester framework
- Comprehensive test suite with integration tests
- Professional development configuration (pyproject.toml, linting, etc.)

### Changed
- Fixed operator overloading to use string literals instead of operator objects
- Enhanced truth table evaluation to handle multiple input formats
- Improved error handling and validation throughout the library

### Fixed
- Truth table representation evaluation bugs
- Import system consistency issues
- Operator overloading parameter passing

## [0.1.0] - 2024-XX-XX

### Added
- Initial library structure and architecture
- Basic Boolean function representations
- Factory pattern for function creation
- Space translation utilities
- Documentation framework

### Notes
- This is the first development release
- Many advanced features are still in development
- API is subject to change in future versions
