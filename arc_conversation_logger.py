#!/usr/bin/env python3
"""
Conversation Logger for ARC AGI AI Assistant
Logs all AI interactions for analysis and debugging
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

class ConversationLogger:
    """Manages logging of all AI conversations"""
    
    def __init__(self, log_dir: str = "logs/conversations"):
        """Initialize the conversation logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different loggers for different purposes
        self.setup_loggers()
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_count = 0
        
    def setup_loggers(self):
        """Setup various loggers for different types of logs"""
        
        # Main conversation logger (human-readable)
        self.conversation_logger = self._create_logger(
            'conversation',
            self.log_dir / 'conversations.log',
            max_bytes=10*1024*1024,  # 10MB
            backup_count=10
        )
        
        # Structured JSON logger for analysis
        self.json_logger = self._create_logger(
            'json_conversation',
            self.log_dir / 'conversations.json',
            max_bytes=50*1024*1024,  # 50MB
            backup_count=5,
            formatter=None  # No formatting for JSON
        )
        
        # Function calls logger
        self.function_logger = self._create_logger(
            'functions',
            self.log_dir / 'function_calls.log',
            max_bytes=5*1024*1024,  # 5MB
            backup_count=5
        )
        
        # Verification attempts logger
        self.verification_logger = self._create_logger(
            'verification',
            self.log_dir / 'verification_attempts.log',
            max_bytes=5*1024*1024,  # 5MB
            backup_count=5
        )
        
        # Daily conversation logger
        self.daily_logger = self._create_daily_logger(
            'daily',
            self.log_dir / 'daily'
        )
        
    def _create_logger(self, name: str, filename: Path, max_bytes: int = 10*1024*1024, 
                      backup_count: int = 5, formatter=None) -> logging.Logger:
        """Create a rotating file logger"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create rotating file handler
        handler = RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Set formatter
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def _create_daily_logger(self, name: str, log_dir: Path) -> logging.Logger:
        """Create a daily rotating logger"""
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create timed rotating handler (daily)
        handler = TimedRotatingFileHandler(
            log_dir / 'conversation.log',
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding='utf-8'
        )
        
        handler.suffix = "%Y%m%d"
        
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def log_conversation(self, user_message: str, ai_response: Dict[str, Any], 
                        puzzle_id: Optional[str] = None, 
                        output_grid: Optional[list] = None,
                        metadata: Optional[Dict] = None):
        """Log a complete conversation exchange"""
        
        self.conversation_count += 1
        timestamp = datetime.now().isoformat()
        
        # Create conversation record
        conversation = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "conversation_number": self.conversation_count,
            "puzzle_id": puzzle_id,
            "user_message": user_message,
            "ai_response": ai_response,
            "output_grid_provided": output_grid is not None,
            "metadata": metadata or {}
        }
        
        # Log to human-readable format
        self.conversation_logger.info(
            f"[Session {self.session_id}] Conversation #{self.conversation_count}\n"
            f"  Puzzle: {puzzle_id or 'None'}\n"
            f"  User: {user_message}\n"
            f"  AI: {ai_response.get('message', 'No message')}\n"
            f"  Type: {ai_response.get('type', 'Unknown')}"
        )
        
        # Log to daily log
        self.daily_logger.info(
            f"[{puzzle_id or 'NO_PUZZLE'}] User: {user_message[:100]}... | "
            f"AI: {ai_response.get('message', '')[:100]}..."
        )
        
        # Log to JSON for analysis
        self.json_logger.info(json.dumps(conversation))
        
        # If there was a function call, log it separately
        if ai_response.get('function_call'):
            self.log_function_call(
                function_name=ai_response['function_call'].get('name'),
                parameters=ai_response['function_call'].get('parameters'),
                result=ai_response.get('function_result'),
                puzzle_id=puzzle_id
            )
        
        # If this was a verification attempt, log it
        if 'verification' in ai_response or 'submission_result' in ai_response:
            self.log_verification_attempt(
                puzzle_id=puzzle_id,
                result=ai_response.get('verification') or ai_response.get('submission_result'),
                message=user_message
            )
    
    def log_function_call(self, function_name: str, parameters: Dict, 
                         result: Any, puzzle_id: Optional[str] = None):
        """Log AI function calls"""
        timestamp = datetime.now().isoformat()
        
        self.function_logger.info(
            f"[{timestamp}] Function: {function_name}\n"
            f"  Puzzle: {puzzle_id or 'None'}\n"
            f"  Parameters: {json.dumps(parameters, indent=2)}\n"
            f"  Result: {json.dumps(result, default=str, indent=2)[:500]}"
        )
    
    def log_verification_attempt(self, puzzle_id: str, result: Dict, message: str):
        """Log verification attempts"""
        timestamp = datetime.now().isoformat()
        
        self.verification_logger.info(
            f"[{timestamp}] Verification for {puzzle_id}\n"
            f"  Message: {message}\n"
            f"  Correct: {result.get('correct', False)}\n"
            f"  Accuracy: {result.get('accuracy', 0)*100:.1f}%\n"
            f"  Attempts Remaining: {result.get('attempts_remaining', 'N/A')}\n"
            f"  Feedback: {result.get('feedback', 'None')}"
        )
    
    def log_heuristic_application(self, heuristic_id: str, heuristic_name: str,
                                 puzzle_id: str, success: bool, confidence: float):
        """Log heuristic applications"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "puzzle_id": puzzle_id,
            "heuristic_id": heuristic_id,
            "heuristic_name": heuristic_name,
            "success": success,
            "confidence": confidence
        }
        
        # Log to function logger as heuristics are a type of function
        self.function_logger.info(
            f"[{timestamp}] Heuristic Applied: {heuristic_name} ({heuristic_id})\n"
            f"  Puzzle: {puzzle_id}\n"
            f"  Success: {success}\n"
            f"  Confidence: {confidence:.2f}"
        )
    
    def log_error(self, error_message: str, error_type: str = "general", 
                 puzzle_id: Optional[str] = None, context: Optional[Dict] = None):
        """Log errors that occur during conversations"""
        timestamp = datetime.now().isoformat()
        
        error_log = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "error_type": error_type,
            "error_message": error_message,
            "puzzle_id": puzzle_id,
            "context": context or {}
        }
        
        self.conversation_logger.error(
            f"[ERROR] {error_type}: {error_message}\n"
            f"  Puzzle: {puzzle_id or 'None'}\n"
            f"  Context: {json.dumps(context or {}, indent=2)}"
        )
        
        # Also log to daily
        self.daily_logger.error(f"[{error_type}] {error_message}")
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about conversations"""
        stats = {
            "session_id": self.session_id,
            "total_conversations": self.conversation_count,
            "session_start": self.session_id,
            "log_directory": str(self.log_dir)
        }
        
        # Count log files
        try:
            stats["log_files"] = {
                "conversation_logs": len(list(self.log_dir.glob("conversations.log*"))),
                "json_logs": len(list(self.log_dir.glob("conversations.json*"))),
                "function_logs": len(list(self.log_dir.glob("function_calls.log*"))),
                "verification_logs": len(list(self.log_dir.glob("verification_attempts.log*"))),
                "daily_logs": len(list((self.log_dir / "daily").glob("*.log*")))
            }
        except:
            stats["log_files"] = "Unable to count"
        
        return stats
    
    def search_conversations(self, query: str, max_results: int = 10) -> list:
        """Search through conversation logs"""
        results = []
        
        try:
            # Search in the current conversation log
            with open(self.log_dir / 'conversations.log', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            current_conv = []
            for line in lines:
                if "Conversation #" in line:
                    if current_conv and query.lower() in ''.join(current_conv).lower():
                        results.append(''.join(current_conv))
                        if len(results) >= max_results:
                            break
                    current_conv = [line]
                else:
                    current_conv.append(line)
            
            # Check last conversation
            if current_conv and query.lower() in ''.join(current_conv).lower():
                results.append(''.join(current_conv))
        
        except Exception as e:
            return [f"Error searching conversations: {e}"]
        
        return results[:max_results]
    
    def export_session_logs(self, export_path: Optional[str] = None) -> str:
        """Export current session logs to a single file"""
        if not export_path:
            export_path = self.log_dir / f"session_{self.session_id}_export.log"
        
        export_path = Path(export_path)
        
        try:
            with open(export_path, 'w', encoding='utf-8') as export_file:
                export_file.write(f"=== Session Export: {self.session_id} ===\n")
                export_file.write(f"Total Conversations: {self.conversation_count}\n\n")
                
                # Read and append conversation log
                if (self.log_dir / 'conversations.log').exists():
                    with open(self.log_dir / 'conversations.log', 'r', encoding='utf-8') as f:
                        export_file.write("=== Conversations ===\n")
                        export_file.write(f.read())
                        export_file.write("\n\n")
                
                # Read and append function calls
                if (self.log_dir / 'function_calls.log').exists():
                    with open(self.log_dir / 'function_calls.log', 'r', encoding='utf-8') as f:
                        export_file.write("=== Function Calls ===\n")
                        export_file.write(f.read())
                        export_file.write("\n\n")
                
                # Read and append verification attempts
                if (self.log_dir / 'verification_attempts.log').exists():
                    with open(self.log_dir / 'verification_attempts.log', 'r', encoding='utf-8') as f:
                        export_file.write("=== Verification Attempts ===\n")
                        export_file.write(f.read())
            
            return str(export_path)
        
        except Exception as e:
            return f"Error exporting logs: {e}"

# Global logger instance
conversation_logger = ConversationLogger()