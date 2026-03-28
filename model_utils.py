import io
import joblib
import pandas as pd


def serialize_model_to_bytes(model_object):
    buffer = io.BytesIO()
    joblib.dump(model_object, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def predict_with_trained_model(model_pipeline, new_df):
    try:
        predictions = model_pipeline.predict(new_df)
        result_df = new_df.copy()
        result_df["预测结果"] = predictions
        return result_df, None
    except Exception as e:
        return None, str(e)


def predict_with_trained_classification_model(model_pipeline, new_df, label_encoder=None):
    try:
        predictions = model_pipeline.predict(new_df)

        if label_encoder is not None:
            decoded_predictions = label_encoder.inverse_transform(predictions)
        else:
            decoded_predictions = predictions

        result_df = new_df.copy()
        result_df["预测结果"] = decoded_predictions

        if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
            proba = model_pipeline.predict_proba(new_df)

            if label_encoder is not None:
                class_names = [str(c) for c in label_encoder.classes_]
            else:
                class_names = [f"类别_{i}" for i in range(proba.shape[1])]

            proba_df = pd.DataFrame(
                proba,
                columns=[f"概率_{class_name}" for class_name in class_names]
            )

            result_df = pd.concat(
                [result_df.reset_index(drop=True), proba_df.reset_index(drop=True)],
                axis=1
            )

            result_df["预测置信度"] = proba.max(axis=1)

        return result_df, None

    except Exception as e:
        return None, str(e)

