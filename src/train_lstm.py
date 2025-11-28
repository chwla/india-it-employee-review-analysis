import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dot, Dense, Dropout, Bidirectional, BatchNormalization, Concatenate, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import pickle

# --- Configuration ---
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 150
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 25
BATCH_SIZE = 32

def run_training():
    print("Starting Enhanced LSTM Training...")
    
    labeled_path = 'data/processed/train_data_large.csv'
    naukri_path = 'data/processed/naukri_processed.csv'
    
    df_labeled = pd.read_csv(labeled_path).fillna("")
    df_naukri = pd.read_csv(naukri_path).fillna("")
    
    # --- Tokenizer ---
    print("Building vocabulary from industry corpus...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_naukri['clean_text'])
    
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"   Vocabulary size: {vocab_size}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # --- Sequences ---
    print(f"Converting {len(df_labeled)} pairs to sequences...")
    resume_seqs = tokenizer.texts_to_sequences(df_labeled['clean_resume'])
    resume_padded = pad_sequences(resume_seqs, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    jd_seqs = tokenizer.texts_to_sequences(df_labeled['clean_jd'])
    jd_padded = pad_sequences(jd_seqs, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    labels = df_labeled['label'].apply(lambda x: 1.0 if x == 'good_match' else 0.0).values
    
    # --- Enhanced Model Architecture ---
    print("Building Enhanced Siamese Network...")
    
    # Shared layers (removed mask_zero to avoid loading issues)
    shared_embedding = Embedding(
        input_dim=vocab_size, 
        output_dim=EMBEDDING_DIM,
        mask_zero=False  # Changed to avoid serialization issues
    )
    shared_lstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))
    shared_dropout = Dropout(0.3)
    shared_bn = BatchNormalization()
    
    # Resume branch
    input_resume = Input(shape=(MAX_SEQ_LENGTH,), name="resume_input")
    embedded_resume = shared_embedding(input_resume)
    lstm_resume = shared_lstm(embedded_resume)
    dropout_resume = shared_dropout(lstm_resume)
    bn_resume = shared_bn(dropout_resume)
    
    # JD branch
    input_jd = Input(shape=(MAX_SEQ_LENGTH,), name="jd_input")
    embedded_jd = shared_embedding(input_jd)
    lstm_jd = shared_lstm(embedded_jd)
    dropout_jd = shared_dropout(lstm_jd)
    bn_jd = shared_bn(dropout_jd)
    
    # Method 1: Cosine similarity
    cosine_sim = Dot(axes=1, normalize=True, name='cosine_similarity')([bn_resume, bn_jd])
    
    # Method 2: Concatenate + Dense layers
    concatenated = Concatenate()([bn_resume, bn_jd])
    dense1 = Dense(64, activation='relu')(concatenated)
    dropout_dense = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu')(dropout_dense)
    semantic_sim = Dense(1, activation='sigmoid', name='semantic_similarity')(dense2)
    
    # Reshape and combine
    cosine_reshaped = Reshape((1,))(cosine_sim)
    combined = Concatenate()([cosine_reshaped, semantic_sim])
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[input_resume, input_jd], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # --- Training ---
    print("\nTraining Model...")
    history = model.fit(
        [resume_padded, jd_padded], 
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save in new Keras format (better compatibility)
    model.save('models/lstm_model.keras')
    print("Model saved to models/lstm_model.keras")
    
    # Also save as .h5 for backwards compatibility
    try:
        model.save('models/lstm_model.h5')
        print("Model also saved to models/lstm_model.h5")
    except:
        print("Could not save .h5 format (not critical)")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)
    print("Training history saved to models/training_history.csv")
    
    # --- Predictions ---
    print("\nGenerating predictions...")
    predictions = model.predict([resume_padded, jd_padded], verbose=0)
    df_labeled['lstm_score'] = predictions.flatten()
    
    os.makedirs('results', exist_ok=True)
    df_labeled.to_csv('results/lstm_predictions.csv', index=False)
    print("Predictions saved to results/lstm_predictions.csv")
    
    # --- Quick Evaluation ---
    from sklearn.metrics import roc_auc_score, classification_report
    
    y_true = (labels > 0.5).astype(int)
    y_pred = (predictions.flatten() > 0.5).astype(int)
    
    auc = roc_auc_score(y_true, predictions.flatten())
    
    print("\n" + "="*50)
    print("LSTM MODEL PERFORMANCE")
    print("="*50)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Poor Match', 'Good Match']))
    print("="*50)

if __name__ == "__main__":
    run_training()
