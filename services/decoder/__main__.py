"""Decoder service entry point"""

import asyncio

from .decoder import main

if __name__ == "__main__":
    asyncio.run(main())
