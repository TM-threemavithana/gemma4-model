from src.adapters.gemma import GemmaAdapter
from src.adapters.whisper import WhisperAdapter
from src.core.agent import GemmaAgent

# Global singletons to be shared across the API and the bridges
gemma_adapter = GemmaAdapter()
whisper_adapter = WhisperAdapter()
agent = GemmaAgent(gemma_adapter, whisper_adapter)
