from pathlib import Path
import torch


def get_contrast_mask(forget_importances, retain_importances, k, part=None):
    # keep top k (%) parameters
    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # import pdb; pdb.set_trace()
    if part == 'vision':
        # forget_importances = {k: v for k, v in forget_importances.items() if 'vision' in k}
        # retain_importances = {k: v for k, v in retain_importances.items() if 'vision' in k}
        all_elements_forget = torch.cat([tensor.flatten() for key, tensor in forget_importances.items() if 'visual' in key])
        all_elements_retain = torch.cat([tensor.flatten() for key, tensor in retain_importances.items() if 'visual' in key])
    if part == None:
        all_elements_forget = torch.cat([tensor.flatten() for tensor in forget_importances.values()])
        all_elements_retain = torch.cat([tensor.flatten() for tensor in retain_importances.values()])

    # for k in threshold_list:
    k_percentile_value = torch.kthvalue(all_elements_forget, int((1 - k) * all_elements_forget.numel()))[0]
    binary_mask_forget = (all_elements_forget >= k_percentile_value).float()
    
    k_percentile_value = torch.kthvalue(all_elements_retain, int((1 - k) * all_elements_retain.numel()))[0]
    binary_mask_retain = (all_elements_retain >= k_percentile_value).float()

    del all_elements_forget, all_elements_retain
    binary_mask_contrast = torch.relu(binary_mask_forget - binary_mask_retain)
    # import pdb; pdb.set_trace()
    # make the mask dictionary
    mask_dict = {}
    start_index = 0
    with torch.no_grad():
        for name in forget_importances:
            params = forget_importances[name]
            num_elements = params.numel()
            if part == 'vision' and 'visual' not in name:
                mask_dict[name] = torch.zeros_like(params)
                # mask_dict[name] = binary_mask_contrast[start_index: start_index + num_elements].reshape(params.shape)
            else:
                mask_dict[name] = binary_mask_contrast[start_index: start_index + num_elements].reshape(params.shape)
                start_index += num_elements
    return mask_dict



if __name__ == '__main__':

    mask_root = Path('clip/mask/')
    model = 'ViT-B-32' 
    # model = 'convnext_base'
    # part = 'vision'
    part = None
    num_shards = 1
    if part:
        forget_importances = torch.load(f'/home/eegrad/.../unlearn/open_clip/src/importance_mask_{model}/forget_importances_shards_{num_shards}_{part}.pt', map_location='cpu')
        retain_importances = torch.load(f'/home/eegrad/.../unlearn/open_clip/src/importance_mask_{model}/retain_importances_shards_{num_shards}_{part}.pt', map_location='cpu')
    else:
        forget_importances = torch.load(mask_root/f'importance_mask_{model}/forget_importances_shards_{num_shards}.pt', map_location='cpu')
        retain_importances = torch.load(mask_root/f'importance_mask_{model}/retain_importances_shards_{num_shards}.pt', map_location='cpu')
    import pdb; pdb.set_trace()



    all_elements_forget = torch.cat([tensor.flatten() for tensor in forget_importances.values()])
    all_elements_retain = torch.cat([tensor.flatten() for tensor in retain_importances.values()])
    nan_percentage_forget = torch.isnan(all_elements_forget).sum() / len(all_elements_forget)
    nan_percentage_retain = torch.isnan(all_elements_retain).sum() / len(all_elements_retain)
    print(f"nan_percentage_forget: {nan_percentage_forget}")
    print(f"nan_percentage_retain: {nan_percentage_retain}")
    print(f'Forget importance: min {all_elements_forget.min()}, max {all_elements_forget.max()}, mean {all_elements_forget.mean()}, std {all_elements_forget.std()}')
    print(f'Retain importance: min {all_elements_retain.min()}, max {all_elements_retain.max()}, mean {all_elements_retain.mean()}, std {all_elements_retain.std()}')



    contrast_mask = get_contrast_mask(forget_importances, retain_importances, 0.5)
    all_elements_mask = torch.cat([tensor.flatten() for tensor in contrast_mask.values()])
    print(f'Percentage of elements that are masked: {100 * all_elements_mask.sum() / all_elements_mask.numel()}')
    mask=contrast_mask



    print('\n Rank by percentage')
    percentage_masked = {}
    for key in mask:
        percentage_masked[key] = 100 * mask[key].sum() / mask[key].numel()
    percentage_masked = {k: v for k, v in sorted(percentage_masked.items(), key=lambda item: item[1], reverse=True)}
    for key in percentage_masked:
        print(f'Layer {key}: {percentage_masked[key]}%')


    print('\n Rank by number')
    number_maksed = {}
    for key in mask:
        number_maksed[key] = mask[key].sum()
    number_maksed = {k: v for k, v in sorted(number_maksed.items(), key=lambda item: item[1], reverse=True)}
    for key in number_maksed:
        print(f'Layer {key}: {number_maksed[key]}, {percentage_masked[key]}')
        

    import pdb; pdb.set_trace()
    print()