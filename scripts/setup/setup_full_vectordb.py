#!/usr/bin/env python3
"""
Setup Full BookCorpus Vector Database with Progress Tracking

This script helps you set up the full BookCorpus vector database for use with
the standard RAG baseline. It provides options to build the database incrementally
or all at once, with comprehensive progress tracking and time estimation.

Usage:
    python setup_full_vectordb.py --build-small    # Build with 10k documents (quick test)
    python setup_full_vectordb.py --build-medium  # Build with 50k documents (medium)
    python setup_full_vectordb.py --build-full    # Build with full dataset (long)
    python setup_full_vectordb.py --test          # Test existing database
    python setup_full_vectordb.py --stats          # Show database statistics
    python setup_full_vectordb.py --monitor        # Monitor running build progress
"""

import sys
import os
from pathlib import Path
import argparse
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import signal

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Add VectorDB to path
sys.path.append(os.path.join(project_root, 'VectorDB'))

from VectorDB.vectordb_manager import VectorDBManager


class ProgressTracker:
    """Track and display progress for database building operations."""
    
    def __init__(self, total_documents: int, db_path: str):
        self.total_documents = total_documents
        self.db_path = db_path
        self.start_time = time.time()
        self.last_update = time.time()
        self.processed_documents = 0
        self.progress_file = os.path.join(db_path, "build_progress.json")
        self.running = True
        
        # Create progress file
        os.makedirs(db_path, exist_ok=True)
        self._save_progress()
        
    def _save_progress(self):
        """Save current progress to file."""
        progress_data = {
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'start_time': self.start_time,
            'last_update': time.time(),
            'percentage': (self.processed_documents / self.total_documents) * 100 if self.total_documents > 0 else 0
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def update_progress(self, processed: int):
        """Update progress and display status."""
        self.processed_documents = processed
        current_time = time.time()
        
        # Update every 30 seconds or when significant progress is made
        if current_time - self.last_update > 30 or processed % 1000 == 0:
            self._display_progress()
            self._save_progress()
            self.last_update = current_time
    
    def _display_progress(self):
        """Display current progress with time estimates."""
        if self.total_documents == 0:
            return
            
        percentage = (self.processed_documents / self.total_documents) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.processed_documents > 0:
            # Calculate ETA
            docs_per_second = self.processed_documents / elapsed_time
            remaining_docs = self.total_documents - self.processed_documents
            eta_seconds = remaining_docs / docs_per_second if docs_per_second > 0 else 0
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            
            print(f"\n📊 PROGRESS UPDATE")
            print(f"   Documents processed: {self.processed_documents:,} / {self.total_documents:,} ({percentage:.1f}%)")
            print(f"   Elapsed time: {self._format_time(elapsed_time)}")
            print(f"   Processing rate: {docs_per_second:.1f} docs/sec")
            print(f"   Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Time remaining: {self._format_time(eta_seconds)}")
        else:
            print(f"\n📊 PROGRESS UPDATE")
            print(f"   Starting database build...")
            print(f"   Total documents to process: {self.total_documents:,}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def finish(self):
        """Mark progress as complete."""
        self.running = False
        elapsed_time = time.time() - self.start_time
        print(f"\n✅ BUILD COMPLETE")
        print(f"   Total documents processed: {self.processed_documents:,}")
        print(f"   Total time: {self._format_time(elapsed_time)}")
        print(f"   Average rate: {self.processed_documents/elapsed_time:.1f} docs/sec")
        
        # Clean up progress file
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except Exception:
            pass


def load_progress(db_path: str) -> Optional[Dict[str, Any]]:
    """Load progress from file if it exists."""
    progress_file = os.path.join(db_path, "build_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def monitor_progress(db_path: str = "./full_bookcorpus_db"):
    """Monitor the progress of a running database build."""
    print("🔍 Monitoring database build progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            progress = load_progress(db_path)
            if progress:
                percentage = progress.get('percentage', 0)
                processed = progress.get('processed_documents', 0)
                total = progress.get('total_documents', 0)
                start_time = progress.get('start_time', time.time())
                
                elapsed = time.time() - start_time
                if processed > 0:
                    docs_per_second = processed / elapsed
                    remaining = total - processed
                    eta_seconds = remaining / docs_per_second if docs_per_second > 0 else 0
                    eta = datetime.now() + timedelta(seconds=eta_seconds)
                    
                    print(f"\r📊 Progress: {processed:,}/{total:,} ({percentage:.1f}%) | "
                          f"Rate: {docs_per_second:.1f} docs/sec | "
                          f"ETA: {eta.strftime('%H:%M:%S')}", end="", flush=True)
                else:
                    print(f"\r📊 Starting build... {total:,} documents to process", end="", flush=True)
            else:
                print(f"\r📊 No active build detected", end="", flush=True)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")


def build_small_database(db_path: str = "./full_bookcorpus_db") -> bool:
    """Build a small database for testing (10k documents)."""
    print("🔧 Building small database (10k documents)...")
    
    # Initialize progress tracker
    tracker = ProgressTracker(10000, db_path)
    
    # Create progress callback
    def progress_callback(processed_documents, total_documents, total_chunks, errors):
        tracker.update_progress(processed_documents)
    
    manager = VectorDBManager(
        db_path=db_path,
        collection_name="full_bookcorpus",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100
    )
    
    # Set progress callback on the builder
    manager.builder.progress_callback = progress_callback
    
    # Override the process_dataset to use fewer documents
    original_process = manager.builder.process_dataset
    
    def limited_process_dataset(dataset_name, num_documents, min_text_length):
        # Limit to 10k documents for small build
        limited_docs = min(num_documents, 10000)
        tracker.total_documents = limited_docs
        
        result = original_process(dataset_name, limited_docs, min_text_length)
        tracker.finish()
        return result
    
    manager.builder.process_dataset = limited_process_dataset
    
    success = manager.initialize_database()
    if success:
        print("✅ Small database built successfully")
        return True
    else:
        print("❌ Small database build failed")
        return False


def build_medium_database(db_path: str = "./full_bookcorpus_db") -> bool:
    """Build a medium database (50k documents)."""
    print("🔧 Building medium database (50k documents)...")
    
    # Initialize progress tracker
    tracker = ProgressTracker(50000, db_path)
    
    # Create progress callback
    def progress_callback(processed_documents, total_documents, total_chunks, errors):
        tracker.update_progress(processed_documents)
    
    manager = VectorDBManager(
        db_path=db_path,
        collection_name="full_bookcorpus",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100
    )
    
    # Set progress callback on the builder
    manager.builder.progress_callback = progress_callback
    
    # Override the process_dataset to use medium number of documents
    original_process = manager.builder.process_dataset
    
    def limited_process_dataset(dataset_name, num_documents, min_text_length):
        # Limit to 50k documents for medium build
        limited_docs = min(num_documents, 50000)
        tracker.total_documents = limited_docs
        
        result = original_process(dataset_name, limited_docs, min_text_length)
        tracker.finish()
        return result
    
    manager.builder.process_dataset = limited_process_dataset
    
    success = manager.initialize_database()
    if success:
        print("✅ Medium database built successfully")
        return True
    else:
        print("❌ Medium database build failed")
        return False


def build_full_database(db_path: str = "./full_bookcorpus_db", force_rebuild: bool = False) -> bool:
    """Build the full database (entire BookCorpus dataset)."""
    print("🔧 Building full database (entire BookCorpus dataset)...")
    print("⚠️  This will take several hours and require significant disk space")
    
    # Get total document count first
    try:
        from datasets import load_dataset
        dataset = load_dataset("rojagtap/bookcorpus", split='train')
        total_docs = len(dataset)
        print(f"📊 Total documents in BookCorpus: {total_docs:,}")
    except Exception as e:
        print(f"⚠️  Could not determine total document count: {e}")
        total_docs = 11000  # Approximate fallback
    
    # Initialize progress tracker
    tracker = ProgressTracker(total_docs, db_path)
    
    # Create progress callback
    def progress_callback(processed_documents, total_documents, total_chunks, errors):
        tracker.update_progress(processed_documents)
    
    response = input("Continue with full database build? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return False
    
    manager = VectorDBManager(
        db_path=db_path,
        collection_name="full_bookcorpus",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100
    )
    
    # Set progress callback on the builder
    manager.builder.progress_callback = progress_callback
    
    success = manager.build_full_database(force_rebuild=force_rebuild)
    if success:
        tracker.finish()
        print("✅ Full database built successfully")
        return True
    else:
        print("❌ Full database build failed")
        return False


def test_database(db_path: str = "./full_bookcorpus_db") -> bool:
    """Test the existing database."""
    print("🧪 Testing database...")
    
    manager = VectorDBManager(
        db_path=db_path,
        collection_name="full_bookcorpus"
    )
    
    if not manager.is_database_ready():
        print("❌ Database not ready")
        return False
    
    success = manager.test_database()
    if success:
        print("✅ Database tests passed")
        return True
    else:
        print("❌ Database tests failed")
        return False


def show_stats(db_path: str = "./full_bookcorpus_db"):
    """Show database statistics."""
    print("📊 Database Statistics")
    print("=" * 50)
    
    manager = VectorDBManager(
        db_path=db_path,
        collection_name="full_bookcorpus"
    )
    
    stats = manager.get_database_stats()
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    for key, value in stats.items():
        print(f"{key}: {value}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup Full BookCorpus Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build small database for testing (10k documents)
  python setup_full_vectordb.py --build-small
  
  # Build medium database (50k documents)
  python setup_full_vectordb.py --build-medium
  
  # Build full database (entire dataset)
  python setup_full_vectordb.py --build-full
  
  # Test existing database
  python setup_full_vectordb.py --test
  
  # Show database statistics
  python setup_full_vectordb.py --stats
        """
    )
    
    parser.add_argument("--build-small", action="store_true", help="Build small database (10k docs)")
    parser.add_argument("--build-medium", action="store_true", help="Build medium database (50k docs)")
    parser.add_argument("--build-full", action="store_true", help="Build full database (entire dataset)")
    parser.add_argument("--test", action="store_true", help="Test existing database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--monitor", action="store_true", help="Monitor running build progress")
    parser.add_argument("--db-path", type=str, default="./full_bookcorpus_db", help="Database path")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if database exists")
    
    args = parser.parse_args()
    
    if args.build_small:
        success = build_small_database(args.db_path)
        if success:
            show_stats(args.db_path)
    
    elif args.build_medium:
        success = build_medium_database(args.db_path)
        if success:
            show_stats(args.db_path)
    
    elif args.build_full:
        success = build_full_database(args.db_path, force_rebuild=args.force_rebuild)
        if success:
            show_stats(args.db_path)
    
    elif args.test:
        test_database(args.db_path)
    
    elif args.stats:
        show_stats(args.db_path)
    
    elif args.monitor:
        monitor_progress(args.db_path)
    
    else:
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python setup_full_vectordb.py --build-small    # Quick test")
        print("  python setup_full_vectordb.py --build-medium   # Medium build")
        print("  python setup_full_vectordb.py --build-full     # Full build")
        print("  python setup_full_vectordb.py --monitor        # Monitor progress")


if __name__ == "__main__":
    main()
