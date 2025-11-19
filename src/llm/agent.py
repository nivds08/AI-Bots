from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from langchain_core.tools import Tool



import os
from openai import OpenAI


class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent with:
    ✓ RAG search over ChromaDB
    ✓ OpenAI LLM responses
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.collection = None
        self.embedding_model = None
        self.tools = []

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    async def initialize(self) -> None:
        """Initialize knowledge base and OpenAI client."""
        await self._setup_knowledge_base()

        # Initialize OpenAI SDK
        self.llm = OpenAI(api_key=self.config.get("api_key", os.getenv("OPENAI_API_KEY")))

        # Setup tools
        self.tools = await self._create_tools()

        self.is_initialized = True

    # -------------------------------------------------------------------------
    # KNOWLEDGE BASE SETUP
    # -------------------------------------------------------------------------
    async def _setup_knowledge_base(self) -> None:
        """Set up ChromaDB + SentenceTransformer embeddings."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import os, hashlib

            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=db_path)

            collection_name = "customer_support_kb"

            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    print(f"Knowledge base already exists with {self.collection.count()} documents")
                    return
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"}
                )

            # Load docs
            knowledge_docs = self._get_customer_support_documents()

            # Model for embeddings
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            docs, metas, ids = [], [], []
            for i, d in enumerate(knowledge_docs):
                doc_id = f"doc_{i}_{hashlib.md5(d['content'].encode()).hexdigest()[:8]}"
                docs.append(d["content"])
                metas.append({
                    "category": d["category"],
                    "title": d["title"],
                    "doc_id": doc_id
                })
                ids.append(doc_id)

            self.collection.add(documents=docs, metadatas=metas, ids=ids)

            print("Knowledge base ready.")

        except Exception as e:
            print(f"Error setting up knowledge base: {e}")
            raise

    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """Returns predefined knowledge base documents."""
        return [
            {"title": "Return Policy Overview", "category": "returns",
             "content": "We offer a 30-day return policy for all products purchased from our store..."},
            {"title": "Return Process Steps", "category": "returns",
             "content": "To initiate a return: log into your account..."},
            {"title": "Non-Returnable Items", "category": "returns",
             "content": "Items such as personalized products, perishable goods..."},
            {"title": "Shipping Methods and Times", "category": "shipping",
             "content": "Standard shipping 5-7 days, Express 2-3 days..."},
            {"title": "International Shipping", "category": "shipping",
             "content": "We ship to 50+ countries..."},
            {"title": "Order Tracking", "category": "shipping",
             "content": "Track orders via email link..."},
            {"title": "Contact Information", "category": "support",
             "content": "Call 1-800-HELP-NOW or email support@company.com..."},
            {"title": "Response Times", "category": "support",
             "content": "Live chat immediate, phone under 3 minutes..."},
            {"title": "Product Warranty", "category": "warranty",
             "content": "Electronics 1-year warranty..."},
            {"title": "Technical Support", "category": "technical",
             "content": "Troubleshooting available Mon-Fri..."},
            {"title": "Account Management", "category": "account",
             "content": "Update profile, view orders..."},
            {"title": "Order Modifications", "category": "orders",
             "content": "Can modify within 1 hour..."},
            {"title": "Payment Methods", "category": "payment",
             "content": "Visa, Mastercard, PayPal, Klarna..."},
            {"title": "Billing and Invoices", "category": "billing",
             "content": "Billing occurs when order ships..."},
            {"title": "Product Availability", "category": "products",
             "content": "In Stock, Limited Stock labels..."},
            {"title": "Size and Fit Guide", "category": "products",
             "content": "Use size charts; size up if unsure..."},
        ]

    # -------------------------------------------------------------------------
    # TOOLS
    # -------------------------------------------------------------------------
    async def _create_tools(self) -> List[Tool]:
        """Only one tool: RAG search."""
        return [
            Tool(
                name="knowledge_search",
                description="Search the customer support knowledge base",
                func=self._rag_search
            )
        ]

    # -------------------------------------------------------------------------
    # RAG SEARCH
    # -------------------------------------------------------------------------
    async def _rag_search(self, query: str) -> str:
        """Runs semantic search over ChromaDB."""
        if not self.collection:
            return "Knowledge base not available."

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if not docs:
                return "I couldn't find relevant information."

            parts = []
            for d, m in zip(docs, metas):
                parts.append(f"**{m['title']}** (Category: {m['category']})\n{d}")

            return "\n\n".join(parts)

        except Exception as e:
            return f"RAG error: {e}"

    # -------------------------------------------------------------------------
    # MAIN QUERY PROCESSING
    # -------------------------------------------------------------------------
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Combines:
        1. RAG search
        2. OpenAI summarization
        """
        rag_result = await self._rag_search(text)

        # Now ask OpenAI to summarize & answer properly
        try:
            response = self.llm.chat.completions.create(
                model=self.config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a customer support AI."},
                    {"role": "user",
                     "content": f"User question: {text}\n\nRelevant info:\n{rag_result}\n\nGive a clear final answer."}
                ],
                temperature=0.4
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"LLM error: {e}"

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    async def cleanup(self) -> None:
        self.llm = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False
