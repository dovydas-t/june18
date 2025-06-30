import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class EncodingManager:
    """Handles categorical variable encoding."""
    
    def __init__(self, ui):
        self.ui = ui
        self.encoders = {}
    
    def encode_categorical(self, pipeline):
        """Enhanced categorical encoding."""
        categorical_cols = pipeline.X_train.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            self.ui.print("✅ No categorical variables to encode")
            return
        
        self.ui.print(f"🏷️ Encoding {len(categorical_cols)} categorical variables")
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            
            # Fit on training data
            pipeline.X_train[col] = encoder.fit_transform(pipeline.X_train[col].astype(str))
            
            # Transform validation data (handle unseen categories)
            val_encoded = self._safe_label_encode(pipeline.X_test[col].astype(str), encoder)
            pipeline.X_test[col] = val_encoded
            
            # Transform submission data if available
            if pipeline.X_submission is not None and col in pipeline.X_submission.columns:
                sub_encoded = self._safe_label_encode(pipeline.X_submission[col].astype(str), encoder)
                pipeline.X_submission[col] = sub_encoded
            
            self.encoders[col] = encoder
    
    def _safe_label_encode(self, series, encoder):
        """Safely encode labels, handling unseen categories."""
        encoded = []
        for val in series:
            if val in encoder.classes_:
                encoded.append(encoder.transform([val])[0])
            else:
                # Use the most frequent class as default
                encoded.append(0)
        return encoded