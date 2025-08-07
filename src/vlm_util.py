from datasets import load_dataset
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

def get_dataset(hf_dataset="ytan-ucr/mu_llava_celeb", subset_size=None):
    # load vlm dataset
    dataset = load_dataset(hf_dataset)
    dataset.set_format(type="torch", columns=['image_id', 'image', 'question', 'answer'])
    dataloader = DataLoader(dataset['test'], batch_size=1, shuffle=True)

    if subset_size is not None:
        subset = Subset(dataloader.dataset, indices=range(subset_size))
        dataloader = DataLoader(subset, batch_size=1, shuffle=True)
    return dataloader


def eval_vlm(vlm_model, vlm_processor, dataloader):
    total_count = 0
    total_correct = 0

    for instance in tqdm(dataloader):
        answer = instance['answer'][0]
        user_prompt = instance['question'][0]
        image = instance['image']
        
        # Prepare input
        prompt = f"USER: <image>\n{user_prompt} ANSER:"
        inputs = vlm_processor(text=prompt, images=image, return_tensors="pt").to(vlm_model.device)
        
        # Generate response
        generate_ids = vlm_model.generate(**inputs, max_new_tokens=50)
        output_text = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Collect result
        if output_text.lower().strip() == answer.lower().strip() or answer.lower().strip() in output_text.lower().strip():
            total_correct += 1
        total_count += 1

    acc = total_correct / total_count
    return acc 