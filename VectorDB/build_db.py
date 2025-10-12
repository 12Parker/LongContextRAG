"""
ChromaDB Vector Database Builder for BookCorpus Dataset
This script loads the BookCorpus dataset and builds a vector database using ChromaDB
with proper error handling, progress tracking, and efficient chunking.
"""

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from tqdm import tqdm
import time
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectordb_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    """Builds a ChromaDB vector database from the BookCorpus dataset"""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "bookcorpus",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 100
    ):
        """
        Initialize the VectorDB Builder
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model to use
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            batch_size: Number of chunks to process in each batch
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        logger.info(f"Initializing ChromaDB at {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.collection = None
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_chunks': 0,
            'skipped_documents': 0,
            'errors': 0
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def create_or_reset_collection(self, reset: bool = False):
        """
        Create or get the ChromaDB collection
        
        Args:
            reset: If True, delete existing collection and create new one
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.info(f"No existing collection to delete: {e}")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        logger.info(f"Collection '{self.collection_name}' ready")
    
    def process_dataset(
        self,
        dataset_name: str = "rojagtap/bookcorpus",
        num_documents: int = 5000,
        min_text_length: int = 100
    ):
        """
        Process the dataset and build the vector database
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            num_documents: Number of documents to process
            min_text_length: Minimum text length to process (characters)
        """
        try:
            # Load dataset
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split='train')
            
            # Select subset
            total_available = len(dataset)
            num_documents = min(num_documents, total_available)
            dataset_subset = dataset.select(range(num_documents))
            
            logger.info(f"Processing {num_documents} documents from {total_available} available")
            self.stats['total_documents'] = num_documents
            
            # Process documents
            chunk_id = 0
            batch_documents = []
            batch_ids = []
            batch_metadatas = []
            
            # Progress bar for documents
            with tqdm(total=num_documents, desc="Processing documents", unit="doc") as pbar:
                for doc_idx, example in enumerate(dataset_subset):
                    try:
                        text = example['text']
                        
                        # Skip short texts
                        if len(text) < min_text_length:
                            self.stats['skipped_documents'] += 1
                            pbar.update(1)
                            continue
                        
                        # Chunk the text
                        chunks = self.chunk_text(text)
                        
                        # Add chunks to batch
                        for chunk_idx, chunk in enumerate(chunks):
                            batch_documents.append(chunk)
                            batch_ids.append(f"chunk_{chunk_id}")
                            batch_metadatas.append({
                                "doc_id": doc_idx,
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks),
                                "text_length": len(chunk)
                            })
                            chunk_id += 1
                        
                        # Add batch to ChromaDB when it reaches batch_size
                        if len(batch_documents) >= self.batch_size:
                            self._add_batch(batch_documents, batch_ids, batch_metadatas)
                            batch_documents = []
                            batch_ids = []
                            batch_metadatas = []
                        
                        self.stats['processed_documents'] += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {doc_idx}: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                        continue
            
            # Add remaining documents in the last batch
            if batch_documents:
                self._add_batch(batch_documents, batch_ids, batch_metadatas)
            
            self.stats['total_chunks'] = chunk_id
            self._print_stats()
            
        except Exception as e:
            logger.error(f"Fatal error during dataset processing: {e}")
            raise
    
    def _add_batch(self, documents: List[str], ids: List[str], metadatas: List[dict]):
        """
        Add a batch of documents to ChromaDB with error handling
        
        Args:
            documents: List of document texts
            ids: List of document IDs
            metadatas: List of metadata dictionaries
        """
        try:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.debug(f"Added batch of {len(documents)} chunks")
        except Exception as e:
            logger.error(f"Error adding batch to ChromaDB: {e}")
            # Try adding documents one by one to identify problematic ones
            for doc, doc_id, metadata in zip(documents, ids, metadatas):
                try:
                    self.collection.add(
                        documents=[doc],
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to add document {doc_id}: {inner_e}")
                    self.stats['errors'] += 1
    
    def _print_stats(self):
        """Print processing statistics"""
        logger.info("\n" + "="*50)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*50)
        logger.info(f"Total documents: {self.stats['total_documents']}")
        logger.info(f"Processed documents: {self.stats['processed_documents']}")
        logger.info(f"Skipped documents: {self.stats['skipped_documents']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info(f"Collection size: {self.collection.count()}")
        logger.info("="*50 + "\n")
    
    def query(self, query_text: str, n_results: int = 5) -> dict:
        """
        Query the vector database
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            Query results dictionary
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Run create_or_reset_collection first.")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    
    def print_query_results(self, results: dict):
        """Pretty print query results"""
        print("\n" + "="*70)
        print("QUERY RESULTS")
        print("="*70)
        
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            print(f"\n[Result {i+1}] Distance: {distance:.4f}")
            print(f"Metadata: {metadata}")
            print(f"Text: {doc[:300]}...")
            print("-"*70)


def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Build a ChromaDB vector database from BookCorpus dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5000 documents (default)
  python build_vectordb.py
  
  # Test with 10000 documents
  python build_vectordb.py --num-docs 10000
  
  # Build full database
  python build_vectordb.py --full
  
  # Custom configuration
  python build_vectordb.py --num-docs 50000 --collection-name my_collection --chunk-size 1000
        """
    )
    
    parser.add_argument(
        '--num-docs',
        type=int,
        default=5000,
        help='Number of documents to process (default: 5000)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Process the entire dataset (overrides --num-docs)'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default='./chroma_db',
        help='Path to store ChromaDB data (default: ./chroma_db)'
    )
    
    parser.add_argument(
        '--collection-name',
        type=str,
        default='bookcorpus',
        help='Name of the ChromaDB collection (default: bookcorpus)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Number of words per chunk (default: 500)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Number of words to overlap between chunks (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of chunks to process in each batch (default: 100)'
    )
    
    parser.add_argument(
        '--no-reset',
        action='store_true',
        help='Do not reset existing collection (append mode)'
    )
    
    parser.add_argument(
        '--skip-queries',
        action='store_true',
        help='Skip test queries at the end'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='Minimum text length in characters to process (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Determine number of documents
    if args.full:
        NUM_DOCUMENTS = None  # Process all documents
        print("\nWARNING: FULL DATABASE MODE - This will process the entire dataset!")
        print("This may take several hours and require significant disk space.")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    else:
        NUM_DOCUMENTS = args.num_docs
    
    # Configuration
    DB_PATH = args.db_path
    COLLECTION_NAME = args.collection_name
    
    print("\n" + "="*70)
    print("ChromaDB Vector Database Builder")
    print("="*70)
    print(f"Documents to process: {NUM_DOCUMENTS if NUM_DOCUMENTS else 'ALL'}")
    print(f"Database path: {DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Chunk size: {args.chunk_size} words")
    print(f"Chunk overlap: {args.chunk_overlap} words")
    print(f"Batch size: {args.batch_size} chunks")
    print("="*70)
    
    # Initialize builder
    builder = VectorDBBuilder(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )
    
    # Create collection
    reset_collection = not args.no_reset
    builder.create_or_reset_collection(reset=reset_collection)
    
    # Process dataset
    start_time = time.time()
    
    if NUM_DOCUMENTS:
        builder.process_dataset(
            dataset_name="rojagtap/bookcorpus",
            num_documents=NUM_DOCUMENTS,
            min_text_length=args.min_length
        )
    else:
        # Process all documents
        logger.info("Loading full dataset...")
        dataset = load_dataset("rojagtap/bookcorpus", split='train')
        total_docs = len(dataset)
        logger.info(f"Processing all {total_docs} documents")
        
        builder.process_dataset(
            dataset_name="rojagtap/bookcorpus",
            num_documents=total_docs,
            min_text_length=args.min_length
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Test queries (optional)
    if not args.skip_queries:
        print("\n" + "="*70)
        print("Testing Vector Database with Sample Queries")
        print("="*70)
        
        test_queries = [
            "A young girl discovers she has magical powers",
            "Science fiction adventure in space",
            "Mystery and detective story"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = builder.query(query, n_results=3)
            builder.print_query_results(results)
            print("\n" + "-"*70)
    else:
        logger.info("Skipping test queries (--skip-queries flag set)")


if __name__ == "__main__":
    main()
