#!/usr/bin/env python3
"""
Monitor Database Build Progress

This script monitors the progress of a running database build and displays
real-time statistics including ETA, processing rate, and completion percentage.

Usage:
    python monitor_build.py
    python monitor_build.py --db-path ./my_database
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def load_progress(db_path: str) -> dict:
    """Load progress from file if it exists."""
    progress_file = os.path.join(db_path, "build_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def monitor_progress(db_path: str = "./full_bookcorpus_db"):
    """Monitor the progress of a running database build."""
    print("🔍 Monitoring database build progress...")
    print("Press Ctrl+C to stop monitoring")
    print(f"Database path: {db_path}")
    print("=" * 60)
    
    last_processed = 0
    last_time = time.time()
    
    try:
        while True:
            progress = load_progress(db_path)
            if progress:
                percentage = progress.get('percentage', 0)
                processed = progress.get('processed_documents', 0)
                total = progress.get('total_documents', 0)
                start_time = progress.get('start_time', time.time())
                
                elapsed = time.time() - start_time
                current_time = time.time()
                
                # Calculate processing rate
                if processed > 0:
                    docs_per_second = processed / elapsed
                    remaining = total - processed
                    eta_seconds = remaining / docs_per_second if docs_per_second > 0 else 0
                    eta = datetime.now() + timedelta(seconds=eta_seconds)
                    
                    # Calculate recent processing rate (last 30 seconds)
                    if current_time - last_time > 30:
                        recent_rate = (processed - last_processed) / (current_time - last_time)
                        last_processed = processed
                        last_time = current_time
                    else:
                        recent_rate = docs_per_second
                    
                    # Display progress
                    print(f"\r📊 Progress: {processed:,}/{total:,} ({percentage:.1f}%) | "
                          f"Rate: {docs_per_second:.1f} docs/sec | "
                          f"Recent: {recent_rate:.1f} docs/sec | "
                          f"ETA: {eta.strftime('%H:%M:%S')} | "
                          f"Elapsed: {format_time(elapsed)}", end="", flush=True)
                else:
                    print(f"\r📊 Starting build... {total:,} documents to process", end="", flush=True)
            else:
                print(f"\r📊 No active build detected", end="", flush=True)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")


def show_current_progress(db_path: str = "./full_bookcorpus_db"):
    """Show current progress without continuous monitoring."""
    progress = load_progress(db_path)
    if not progress:
        print("❌ No active build detected")
        return
    
    percentage = progress.get('percentage', 0)
    processed = progress.get('processed_documents', 0)
    total = progress.get('total_documents', 0)
    start_time = progress.get('start_time', time.time())
    
    elapsed = time.time() - start_time
    
    print("📊 CURRENT BUILD PROGRESS")
    print("=" * 40)
    print(f"Documents processed: {processed:,} / {total:,}")
    print(f"Percentage complete: {percentage:.1f}%")
    print(f"Elapsed time: {format_time(elapsed)}")
    
    if processed > 0:
        docs_per_second = processed / elapsed
        remaining = total - processed
        eta_seconds = remaining / docs_per_second if docs_per_second > 0 else 0
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        
        print(f"Processing rate: {docs_per_second:.1f} docs/sec")
        print(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time remaining: {format_time(eta_seconds)}")
    else:
        print("Build just started...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor Database Build Progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor progress continuously
  python monitor_build.py
  
  # Monitor specific database path
  python monitor_build.py --db-path ./my_database
  
  # Show current progress once
  python monitor_build.py --status
        """
    )
    
    parser.add_argument("--db-path", type=str, default="./full_bookcorpus_db", 
                       help="Database path to monitor")
    parser.add_argument("--status", action="store_true", 
                       help="Show current status once instead of continuous monitoring")
    
    args = parser.parse_args()
    
    if args.status:
        show_current_progress(args.db_path)
    else:
        monitor_progress(args.db_path)


if __name__ == "__main__":
    main()
