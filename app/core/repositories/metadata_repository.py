import pickle
import logging

logger = logging.getLogger(__name__)

class MetadataRepository:
    def load_metadata(self, path: str) -> dict:
        with open(path, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Metadata loaded from {path} with {len(metadata['ids'])} documents")
        
        # Validate and normalize metadata
        for i, meta in enumerate(metadata["metadata"]):
            if "type" not in meta:
                meta["type"] = "banan"
            meta["case_summary"] = meta.get("case_summary", "No summary available")
            meta["legal_issues"] = meta.get("legal_issues", "No legal issues specified")
            meta["court_reasoning"] = meta.get("court_reasoning", "No reasoning provided")
            meta["decision"] = meta.get("decision", "No decision available")
            meta["relevant_laws"] = meta.get("relevant_laws", "No laws cited")
            if not metadata["texts"][i]:
                logger.warning(f"Empty text field for document ID {metadata['ids'][i]}")
        
        return metadata