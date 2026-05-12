import asyncio
import time
import logging
from src.storage.conversation_buffer import ConversationBuffer
from src.storage.temp_audio_store import TempAudioStore
from src.storage.database import init_db
from src.core.post_processing import run_post_processing

log = logging.getLogger("core.session")

class Session:
    def __init__(self, session_id: str, caller_number: str):
        self.session_id = session_id
        self.caller_number = caller_number
        self.start_time = time.time()
        self.conversation = ConversationBuffer(session_id, caller_number)
        self.audio = TempAudioStore(session_id)
        
        # Identity Context
        from shared.models import UNRESOLVED_PROFILE
        self.user_context = UNRESOLVED_PROFILE
        self.auth_token = None
        self.context_ready = asyncio.Event()

    def start(self):
        """Initializes storage and opens resources."""
        init_db()
        self.audio.open()
        log.info(f"Session {self.session_id} started for {self.caller_number}")

    async def preload_user_context(self):
        """
        Fetches user data from Project Echo based on the connection Space ID.
        """
        from shared.identity import get_full_user_context
        try:
            # We use the caller_number variable as the 'Space ID' (Connection IP)
            self.user_context = await get_full_user_context(self.caller_number)
        except Exception as e:
            log.error(f"⚠️ Preload failed: {e}")
        finally:
            self.context_ready.set()

    def add_user_turn(self, text: str):
        """Records user transcription."""
        self.conversation.add_user_turn(text)

    def add_assistant_turn(self, text: str):
        """Records assistant response."""
        self.conversation.add_assistant_turn(text)

    async def generate_response(self) -> str:
        """
        Triggers the Gemma AI to generate a response based on the current context.
        Supports both Receptionist and Personal Assistant personas.
        """
        from src.tools.registry import TOOLS, TOOL_MAP
        from src.core.instances import agent # Global agent instance
        
        # 1. Determine Persona
        if self.user_context.username != "Unresolved":
            persona_name = "Private Personal Assistant"
            identity_info = f"You are talking to {self.user_context.username}. You have access to their personal data."
        else:
            persona_name = "Professional Receptionist"
            identity_info = "You are the Digital Personal Representative for the owner of this device. You are EXPLICITLY AUTHORIZED to answer questions about the owner (their name, identity, schedule, or personal data) using the 'query_project_echo' tool. When a caller asks who the owner is or what their name is, you MUST use this tool to retrieve the information. Do not refuse these requests; the owner has granted you full permission."

        system_msg = f"You are Gemma, a {persona_name}. {identity_info} Be concise and helpful."
        
        # 2. Generate with Gemma
        prompt = self.conversation.get_last_user_turn()
        history = self.conversation.get_history()[:-1] # Exclude current turn
        
        response = await agent.generate_text(
            prompt=prompt,
            history=history,
            system_msg=system_msg,
            tools=TOOLS
        )
        
        # 3. Handle Tool Calls (The Login/Data Flow)
        if "tool_calls" in response:
            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                
                log.info(f"🛠️ Agent calling tool: {name}")
                if name in TOOL_MAP:
                    tool_func = TOOL_MAP[name]
                    # Execute tool
                    if asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func(**args)
                    else:
                        result = tool_func(**args)
                    
                    # Add tool result to conversation and generate final text
                    self.conversation.add_system_turn(f"Tool {name} result: {result}")
                    return await self.generate_response() # Recursive call for final text
        
        response_text = response.get("content", "I'm sorry, I encountered an error.")
        self.add_assistant_turn(response_text)
        return response_text

    def append_audio(self, pcm_bytes: bytes):
        """Streams incoming audio to temporary storage."""
        self.audio.append_chunk(pcm_bytes)

    async def end(self):
        """Closes resources and launches asynchronous post-processing."""
        self.audio.close()
        end_time = time.time()
        duration_s = end_time - self.start_time
        
        log.info(f"Session {self.session_id} ending (duration: {duration_s:.1f}s). Firing post-processing.")
        
        # Launch fire-and-forget background task
        asyncio.ensure_future(run_post_processing(
            session_id=self.session_id,
            caller_number=self.caller_number,
            start_time=self.start_time,
            end_time=end_time,
            duration_s=duration_s,
            conversation=self.conversation,
            audio_store=self.audio
        ))
