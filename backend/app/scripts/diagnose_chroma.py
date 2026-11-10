import json
import logging

from app.implementations.rag_implementation import AgenticRAGImplementation

logging.basicConfig(level=logging.INFO)

def main():
    implementation = AgenticRAGImplementation()
    report = implementation.diagnose_vector_store()

    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()