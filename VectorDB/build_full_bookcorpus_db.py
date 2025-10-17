#!/usr/bin/env python3
"""
Build Full BookCorpus Vector Database

This script builds a persistent vector database from the entire BookCorpus dataset
that can be reused across multiple runs without rebuilding.

Key Features:
- Processes the entire BookCorpus dataset
- Creates persistent ChromaDB storage
- Optimized for large-scale processing
- Progress tracking and error handling
- Reusable across multiple sessions

Usage:
    python VectorDB/build_full_bookcorpus_db.py
    python VectorDB/build_full_bookcorpus_db.py --db-path ./full_bookcorpus_db
    python VectorDB/build_full_bookcorpus_db.py --chunk-size 1000 --batch-size 200
"""

import sys
import os
from pathlib import Path
import time
import logging
from typing import Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from build_db import VectorDBBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_bookcorpus_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FullBookCorpusVectorDB:
    """
    Builds and manages the full BookCorpus vector database.
    """
    
    def __init__(self, 
                 db_path: str = "./full_bookcorpus_db",
                 collection_name: str = "full_bookcorpus",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 batch_size: int = 100,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the full BookCorpus vector database builder.
        
        Args:
            db_path: Path to store the persistent database
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for processing
            embedding_model: Embedding model to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        
        # Initialize VectorDBBuilder
        self.vectordb_config = {
            'db_path': db_path,
            'collection_name': collection_name,
            'embedding_model': embedding_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'batch_size': batch_size
        }
        
        self.builder = VectorDBBuilder(**self.vectordb_config)
        self.is_built = False
        
        logger.info(f"Initialized FullBookCorpusVectorDB")
        logger.info(f"Database path: {db_path}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Embedding model: {embedding_model}")
    
    def check_existing_database(self) -> bool:
        """
        Check if the database already exists and has data.
        
        Returns:
            True if database exists and has data, False otherwise
        """
        try:
            # Try to connect to existing database
            from chromadb import PersistentClient
            client = PersistentClient(path=self.db_path)
            
            # Check if collection exists
            try:
                collection = client.get_collection(name=self.collection_name)
                count = collection.count()
                
                if count > 0:
                    logger.info(f"Found existing database with {count} chunks")
                    return True
                else:
                    logger.info("Database exists but is empty")
                    return False
                    
            except Exception:
                logger.info("Collection does not exist")
                return False
                
        except Exception as e:
            logger.info(f"Database does not exist: {e}")
            return False
    
    def build_database(self, force_rebuild: bool = False) -> bool:
        """
        Build the full BookCorpus vector database.
        
        Args:
            force_rebuild: If True, rebuild even if database exists
            
        Returns:
            True if successful, False otherwise
        """
        # Check if database already exists
        if not force_rebuild and self.check_existing_database():
            logger.info("✅ Database already exists and has data")
            logger.info("Use --force-rebuild to rebuild the database")
            self.is_built = True
            return True
        
        logger.info("🔧 Building full BookCorpus vector database...")
        logger.info("⚠️  This will process the entire BookCorpus dataset")
        logger.info("⚠️  This may take several hours and require significant disk space")
        
        start_time = time.time()
        
        try:
            # Create or reset collection
            self.builder.create_or_reset_collection(reset=True)
            
            # Process the entire dataset
            logger.info("Loading full BookCorpus dataset...")
            from datasets import load_dataset
            
            dataset = load_dataset("rojagtap/bookcorpus", split='train')
            total_docs = len(dataset)
            
            logger.info(f"Processing all {total_docs} documents from BookCorpus")
            
            # Process the dataset
            self.builder.process_dataset(
                dataset_name="rojagtap/bookcorpus",
                num_documents=total_docs,
                min_text_length=100
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Database build completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            self.is_built = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build database: {e}")
            return False
    
    def get_database_stats(self) -> dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.is_built and not self.check_existing_database():
            return {'error': 'Database not built or not found'}
        
        try:
            from chromadb import PersistentClient
            client = PersistentClient(path=self.db_path)
            collection = client.get_collection(name=self.collection_name)
            
            count = collection.count()
            
            return {
                'database_path': self.db_path,
                'collection_name': self.collection_name,
                'total_chunks': count,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': self.embedding_model,
                'is_built': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_database(self, test_queries: list = None) -> bool:
        """
        Test the database with sample queries.
        
        Args:
            test_queries: List of test queries (optional)
            
        Returns:
            True if tests pass, False otherwise
        """
        if not self.is_built and not self.check_existing_database():
            logger.error("Database not built or not found")
            return False
        
        if test_queries is None:
            test_queries = [
                "A young girl discovers she has magical powers",
                "Science fiction adventure in space",
                "Mystery and detective story",
                "Romance and love story",
                "Historical fiction and war"
            ]
        
        logger.info("🧪 Testing database with sample queries...")
        
        try:
            for i, query in enumerate(test_queries, 1):
                logger.info(f"Test query {i}: {query}")
                results = self.builder.query(query, n_results=3)
                
                if not results['documents'][0]:
                    logger.warning(f"No results for query: {query}")
                    return False
                
                logger.info(f"✅ Query {i} returned {len(results['documents'][0])} results")
            
            logger.info("✅ All test queries passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False


def main():
    """Main function to build the full BookCorpus vector database."""
    parser = argparse.ArgumentParser(
        description="Build full BookCorpus vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default settings
  python VectorDB/build_full_bookcorpus_db.py
  
  # Build with custom settings
  python VectorDB/build_full_bookcorpus_db.py --db-path ./my_bookcorpus_db --chunk-size 1000
  
  # Force rebuild existing database
  python VectorDB/build_full_bookcorpus_db.py --force-rebuild
  
  # Skip test queries
  python VectorDB/build_full_bookcorpus_db.py --skip-tests
        """
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default='./full_bookcorpus_db',
        help='Path to store the database (default: ./full_bookcorpus_db)'
    )
    
    parser.add_argument(
        '--collection-name',
        type=str,
        default='full_bookcorpus',
        help='Name of the ChromaDB collection (default: full_bookcorpus)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Size of text chunks (default: 500)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Embedding model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild even if database exists'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip test queries after building'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show database statistics (do not build)'
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = FullBookCorpusVectorDB(
        db_path=args.db_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model
    )
    
    if args.stats_only:
        # Show statistics only
        stats = builder.get_database_stats()
        print("\n📊 DATABASE STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    # Build database
    success = builder.build_database(force_rebuild=args.force_rebuild)
    
    if success:
        # Show statistics
        stats = builder.get_database_stats()
        print("\n📊 DATABASE STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test database
        if not args.skip_tests:
            test_success = builder.test_database()
            if test_success:
                print("\n✅ Database build and testing completed successfully!")
            else:
                print("\n⚠️  Database built but testing failed")
        else:
            print("\n✅ Database build completed successfully!")
    else:
        print("\n❌ Database build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
