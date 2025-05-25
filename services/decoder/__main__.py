"""Decoder service entry point"""
from .decoder import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())