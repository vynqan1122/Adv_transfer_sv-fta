#!/usr/bin/env python3
"""Compatibility entry point for the final SV-FCA Table V'."""
if __package__:
    from .summarize_table5_prime import main
else:
    from summarize_table5_prime import main

if __name__ == "__main__":
    main()
