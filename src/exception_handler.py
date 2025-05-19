import traceback
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_errors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("rag_chatbot")

def handle_exception(exc):
    """
    Comprehensive exception handler that logs detailed error information
    and provides useful context for debugging.
    
    Args:
        exc: The exception that was raised
    """
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get exception details
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        
        # Get stack trace
        tb = traceback.format_exc()
        
        # Log to console
        print(f"\n{'='*80}")
        print(f"ERROR OCCURRED AT: {timestamp}")
        print(f"EXCEPTION TYPE: {exc_type}")
        print(f"EXCEPTION MESSAGE: {exc_msg}")
        print(f"\nTRACEBACK:\n{tb}")
        print(f"{'='*80}\n")
        
        # Log to file with more details
        logger.error(f"Exception occurred: {exc_type}: {exc_msg}")
        logger.error(f"Traceback:\n{tb}")
        
        # Add system info to log
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Create error report directory if it doesn't exist
        os.makedirs("error_reports", exist_ok=True)
        
        # Write detailed error report to file
        report_filename = f"error_reports/error_{timestamp.replace(' ', '_').replace(':', '-')}.log"
        with open(report_filename, "w") as f:
            f.write(f"ERROR REPORT - {timestamp}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"EXCEPTION TYPE: {exc_type}\n")
            f.write(f"EXCEPTION MESSAGE: {exc_msg}\n\n")
            f.write(f"TRACEBACK:\n{tb}\n\n")
            f.write(f"SYSTEM INFO:\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Current working directory: {os.getcwd()}\n")
            f.write(f"\n{'='*80}\n")
        
        print(f"Detailed error report written to: {report_filename}")
        
    except Exception as e:
        # If exception handling itself fails, log this too
        print(f"ERROR IN EXCEPTION HANDLER: {str(e)}")
        logger.error(f"Exception in handle_exception: {str(e)}")