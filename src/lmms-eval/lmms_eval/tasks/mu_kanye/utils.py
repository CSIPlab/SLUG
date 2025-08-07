import json


def mu_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def mu_doc_to_text(doc):
    question = doc["question"]
    return f"{question}"


def mu_process_result(doc, result):
    pred = result[0].strip()
    if '.' in pred:
        pred = pred[:-1]
    # if len(pred) > 1:
    #     pred = pred[0]
    answer = doc["answer"]
    metric_name = 'FA'

    return {f"{metric_name}": {"pred": pred, "answer": answer, "image_id": doc["image_id"]}, f"mu_all": {"pred": pred, "answer": answer, "image_id": doc["image_id"]}}


def mu_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip() or result["answer"].lower().strip() in result["pred"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count


def mu_aggregation_result_all(results):
    score = mu_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"image_id": result["image_id"], "prediction": result["pred"]})
    with open("./mu_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for mu_submission ...")

    return score