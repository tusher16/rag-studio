"""
Test dataset for RAG evaluation.
All questions and ground truths are based on the paper:
"A LayoutLMv3-Based Model for Enhanced Relation Extraction in Visually-Rich Documents"
"""

TEST_CASES = [

    # ── Architecture ──────────────────────────────────────────────────────────
    {
        "question": "What backbone model is used in this paper?",
        "ground_truth": "LayoutLMv3 is used as the backbone model."
    },
    {
        "question": "How many parameters does the LayoutLMv3 large model have?",
        "ground_truth": "The LayoutLMv3 large model contains 357 million parameters."
    },
    {
        "question": "What three tasks was LayoutLMv3 pre-trained on?",
        "ground_truth": "LayoutLMv3 was pre-trained on Masked Language Modeling (MLM), Masked Image Modeling (MIM), and Word-Patch Alignment (WPA)."
    },
    {
        "question": "What layer is used for relation prediction between entity pairs?",
        "ground_truth": "An asymmetric bilinear layer is used for fusing entity representations and predicting relations."
    },
    {
        "question": "What is the training objective for the relation prediction head?",
        "ground_truth": "The relation prediction head is trained via the binary cross-entropy objective function."
    },

    # ── Dataset ───────────────────────────────────────────────────────────────
    {
        "question": "What dataset was used to pre-train the LayoutLMv3 model?",
        "ground_truth": "LayoutLMv3 was pre-trained on a subset of 11 million documents from IIT-CDIP Test Collection 1.0, representing 42 million scanned pages."
    },
    {
        "question": "How many samples does the CORD dataset have and what is the train/val/test split?",
        "ground_truth": "CORD has 1,000 annotated samples divided into training/validation/testing sets with the ratio 800/100/100."
    },
    {
        "question": "How many samples does the FUNSD dataset have and what is the train/test split?",
        "ground_truth": "FUNSD comprises 199 fully annotated scanned forms divided into training/testing sets with the ratio 149/50."
    },
    {
        "question": "What type of documents does the CORD dataset contain?",
        "ground_truth": "CORD is a dataset of restaurant receipts, which includes both images and box/text annotations for OCR, as well as multi-level semantic labels for parsing."
    },
    {
        "question": "What is the batch size used during training?",
        "ground_truth": "During training, inputs are grouped into batches of size 2."
    },

    # ── Methods ───────────────────────────────────────────────────────────────
    {
        "question": "What is the Entity Marker technique?",
        "ground_truth": "Entity Marker integrates entity types within the structured input by presenting entity information within the OCR output text, using the format: entity type followed by the entity span, allowing the model to understand potential entity relationships."
    },
    {
        "question": "What is Bounding Boxes Ordering (BBO)?",
        "ground_truth": "Bounding Boxes Ordering orders boxes by their vertical positions to enhance the model's accuracy, mimicking the human method of processing information and aligning with the natural reading order."
    },
    {
        "question": "What is Bounding Boxes Shuffling (BBS)?",
        "ground_truth": "Bounding Boxes Shuffling randomly permutes the bounding boxes in each batch during fine-tuning, introducing artificial noise into positional encoding to force the model to rely more on spatial coordinates rather than token order."
    },
    {
        "question": "What is Layout Concatenation (LC)?",
        "ground_truth": "Layout Concatenation creates a skip-connection between token-level layout information and the RE head, by concatenating normalized absolute coordinates of each token with its associated hidden state before relation prediction."
    },
    {
        "question": "What is the RSF post-processing method?",
        "ground_truth": "RSF (Restriction on the Selection of Fathers) refines predicted relations by selecting a set of predicted parent entities for each child entity, keeping only the parent with the maximum probability above a predefined margin."
    },

    # ── Results ───────────────────────────────────────────────────────────────
    {
        "question": "What is the best F1 score achieved on FUNSD dataset?",
        "ground_truth": "The best F1 score on FUNSD is 90.81, achieved by LayoutLMv3 LARGE with Entity Marker, Bounding Boxes Ordering and RSF (EM + BBO + RSF)."
    },
    {
        "question": "What is the best F1 score achieved on CORD dataset?",
        "ground_truth": "The best F1 score on CORD is 98.60, achieved by LayoutLMv3 LARGE with Entity Marker and Bounding Boxes Ordering (EM + BBO)."
    },
    {
        "question": "What F1 score does the baseline LayoutLMv3 large achieve on FUNSD?",
        "ground_truth": "The baseline LayoutLMv3 large achieves an F1 score of 82.38 on FUNSD."
    },
    {
        "question": "What is the F1 score of GeoLayoutLM on FUNSD and CORD?",
        "ground_truth": "GeoLayoutLM achieves 89.45 on FUNSD and 97.35 on CORD."
    },

    # ── Contributions ─────────────────────────────────────────────────────────
    {
        "question": "What are the two main contributions of this paper?",
        "ground_truth": "1) A methodology that achieves state-of-the-art RE performance without geometric pre-training and with fewer parameters. 2) An extensive ablation study illustrating the effects of different training setups and additional features specific to VRDs."
    },
    {
        "question": "What is the main advantage of the Entity Marker method over punctuation-based methods?",
        "ground_truth": "The Entity Marker method reduces the number of tokens per prediction — with punctuation, 53% of CORD samples required at least two windows, compared to 0% with the proposed method."
    },
    {
        "question": "What does the ablation study show about Layout Concatenation?",
        "ground_truth": "Layout Concatenation does not improve performance and slightly decreases F1 by 0.70% on FUNSD, likely because the layout information is redundant and already available in token embeddings, increasing model complexity and potentially causing overfitting."
    },
]
