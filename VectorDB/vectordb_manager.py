#!/usr/bin/env python3
"""
VectorDB Manager for Persistent Database Operations

This module provides a manager class to handle persistent vector database operations,
including initialization, reuse, and management of the full BookCorpus dataset.

Key Features:
- Persistent database management
- Automatic database detection and reuse
- Configuration management
- Error handling and validation
- Statistics and monitoring

Usage:
    from VectorDB.vectordb_manager import VectorDBManager
    
    manager = VectorDBManager()
    if not manager.is_database_ready():
        manager.build_database()
    
    results = manager.query("your query here")
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from build_db import VectorDBBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manages persistent vector database operations for the BookCorpus dataset.
    """
    
    def __init__(self, 
                 db_path: str = "./full_bookcorpus_db",
                 collection_name: str = "full_bookcorpus",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 batch_size: int = 100,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the VectorDB manager.
        
        Args:
            db_path: Path to the persistent database
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
        self._is_initialized = False
        
        logger.info(f"VectorDBManager initialized")
        logger.info(f"Database path: {db_path}")
        logger.info(f"Collection: {collection_name}")
    
    def is_database_ready(self) -> bool:
        """
        Check if the database is ready for use.
        
        Returns:
            True if database exists and has data, False otherwise
        """
        try:
            # Check if database directory exists
            if not os.path.exists(self.db_path):
                logger.info("Database directory does not exist")
                return False
            
            # Try to connect to the database
            from chromadb import PersistentClient
            client = PersistentClient(path=self.db_path)
            
            # Check if collection exists and has data
            try:
                collection = client.get_collection(name=self.collection_name)
                count = collection.count()
                
                if count > 0:
                    logger.info(f"Database ready with {count} chunks")
                    # Initialize the builder's collection for querying
                    self.builder.collection = collection
                    self._is_initialized = True
                    return True
                else:
                    logger.info("Database exists but is empty")
                    return False
                    
            except Exception as e:
                logger.info(f"Collection does not exist: {e}")
                return False
                
        except Exception as e:
            logger.info(f"Database not ready: {e}")
            return False
    
    def initialize_database(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the database, building it if necessary.
        
        Args:
            force_rebuild: If True, rebuild even if database exists
            
        Returns:
            True if successful, False otherwise
        """
        if not force_rebuild and self.is_database_ready():
            logger.info("✅ Database already ready")
            return True
        
        logger.info("🔧 Initializing database...")
        
        try:
            # Create or reset collection
            self.builder.create_or_reset_collection(reset=True)
            
            # Process the dataset
            logger.info("Processing BookCorpus dataset...")
            self.builder.process_dataset(
                dataset_name="rojagtap/bookcorpus",
                num_documents=50000,  # Start with 50k documents for testing
                min_text_length=100
            )
            
            self._is_initialized = True
            logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    def build_full_database(self, force_rebuild: bool = False) -> bool:
        """
        Build the full BookCorpus database.
        
        Args:
            force_rebuild: If True, rebuild even if database exists
            
        Returns:
            True if successful, False otherwise
        """
        if not force_rebuild and self.is_database_ready():
            logger.info("✅ Full database already ready")
            return True
        
        logger.info("🔧 Building full BookCorpus database...")
        logger.info("⚠️  This will process the entire BookCorpus dataset")
        logger.info("⚠️  This may take several hours")
        
        try:
            # Create or reset collection
            self.builder.create_or_reset_collection(reset=True)
            
            # Process the entire dataset
            from datasets import load_dataset
            dataset = load_dataset("rojagtap/bookcorpus", split='train')
            total_docs = len(dataset)
            
            logger.info(f"Processing all {total_docs} documents from BookCorpus")
            
            self.builder.process_dataset(
                dataset_name="rojagtap/bookcorpus",
                num_documents=total_docs,
                min_text_length=100
            )
            
            self._is_initialized = True
            logger.info("✅ Full database built successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build full database: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the database.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            Query results dictionary
        """
        if not self._is_initialized and not self.is_database_ready():
            raise ValueError("Database not initialized. Call initialize_database() first.")
        
        try:
            # Ensure the collection is properly initialized for querying
            if not hasattr(self.builder, 'collection') or self.builder.collection is None:
                # Get the collection from the existing database
                self.builder.collection = self.builder.client.get_collection(
                    name=self.collection_name
                )
            
            results = self.builder.query(query_text, n_results=n_results)
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {'error': str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.is_database_ready():
            return {'error': 'Database not ready'}
        
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
                'is_ready': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_database(self, test_queries: List[str] = None) -> bool:
        """
        Test the database with sample queries.
        
        Args:
            test_queries: List of test queries (optional)
            
        Returns:
            True if tests pass, False otherwise
        """
        if not self.is_database_ready():
            logger.error("Database not ready")
            return False
        
        if test_queries is None:
            test_queries = [
                "A young girl discovers she has magical powers",
                "Science fiction adventure in space",
                "Mystery and detective story"
            ]
        
        logger.info("🧪 Testing database...")
        
        try:
            for i, query in enumerate(test_queries, 1):
                results = self.query(query, n_results=3)
                
                if 'error' in results:
                    logger.error(f"Query {i} failed: {results['error']}")
                    return False
                
                if not results['documents'][0]:
                    logger.warning(f"No results for query {i}: {query}")
                    return False
                
                logger.info(f"✅ Query {i} returned {len(results['documents'][0])} results")
            
            logger.info("✅ All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False
    
    def save_config(self, config_path: str = "vectordb_config.json") -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                'db_path': self.db_path,
                'collection_name': self.collection_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'batch_size': self.batch_size,
                'embedding_model': self.embedding_model
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, config_path: str = "vectordb_config.json") -> bool:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to load the configuration from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.db_path = config.get('db_path', self.db_path)
            self.collection_name = config.get('collection_name', self.collection_name)
            self.chunk_size = config.get('chunk_size', self.chunk_size)
            self.chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.embedding_model = config.get('embedding_model', self.embedding_model)
            
            # Reinitialize builder with new config
            self.vectordb_config = {
                'db_path': self.db_path,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'batch_size': self.batch_size
            }
            
            self.builder = VectorDBBuilder(**self.vectordb_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False


def main():
    """Main function for testing the VectorDB manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VectorDB Manager")
    parser.add_argument("--db-path", default="./full_bookcorpus_db", help="Database path")
    parser.add_argument("--collection-name", default="full_bookcorpus", help="Collection name")
    parser.add_argument("--test", action="store_true", help="Test the database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--build", action="store_true", help="Build the database")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = VectorDBManager(
        db_path=args.db_path,
        collection_name=args.collection_name
    )
    
    if args.stats:
        # Show statistics
        stats = manager.get_database_stats()
        print("\n📊 DATABASE STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    if args.build:
        # Build database
        success = manager.build_full_database(force_rebuild=args.force_rebuild)
        if success:
            print("✅ Database built successfully")
        else:
            print("❌ Database build failed")
    
    if args.test:
        # Test database
        if manager.is_database_ready():
            success = manager.test_database()
            if success:
                print("✅ Database tests passed")
            else:
                print("❌ Database tests failed")
        else:
            print("❌ Database not ready")


if __name__ == "__main__":
    main()
