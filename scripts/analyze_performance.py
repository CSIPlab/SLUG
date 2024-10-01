import re
from pathlib import Path


def parse_file(file_path):
    """extract info from txt files"""
    iterations = []
    ratios = []
    fgt_acc1 = []
    fgt_acc5 = []
    test_acc1 = []
    test_acc5 = []
    MIA = []
    celeb_top1 = []
    celeb_top5 = []

    pattern = r'iter: (\d+), fgt_acc@1: ([\d\.]+), fgt_acc@5: ([\d\.]+), celeba100@1: ([\d\.]+), celeba100@5: ([\d\.]+), test_acc@1: ([\d\.]+), test_acc@5: ([\d\.]+), MIA: ([\d\.]+)±([\d\.]+)'
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                iterations.append(int(match.groups()[0]))
                fgt_acc1.append(float(match.groups()[1]))
                fgt_acc5.append(float(match.groups()[2]))
                test_acc1.append(float(match.groups()[5]))
                test_acc5.append(float(match.groups()[6]))
                MIA.append((float(match.groups()[7]), float(match.groups()[8])))
                celeb_top1.append(float(match.groups()[3]))
                celeb_top5.append(float(match.groups()[4]))
            else:
                print("No match found")

    return {
        'iterations': sorted(list(set(iterations))),
        'fgt_acc1': fgt_acc1,
        'fgt_acc5': fgt_acc5,
        'test_acc1': test_acc1,
        'test_acc5': test_acc5,
        'MIA': MIA,
        'celeb_top1': celeb_top1,
        'celeb_top5': celeb_top5
    }


def calculate_averages(data_list):
    # Initialize a dictionary to hold the sum of each field
    sums = {}
    # Initialize the result dictionary for averages
    averages = {}
    
    # Process each item in the list
    for data in data_list:
        for key, value in data.items():
            # Handle numeric values
            if isinstance(value, (int, float)):
                if key in sums:
                    sums[key] += value
                else:
                    sums[key] = value
            # Handle tuple values (for example 'MIA')
            elif isinstance(value, tuple) and all(isinstance(num, (int, float)) for num in value):
                if key not in sums:
                    sums[key] = tuple(0 for _ in value)
                sums[key] = tuple(sum(x) for x in zip(sums[key], value))
    
    # Calculate averages
    num_items = len(data_list)
    for key, sum_value in sums.items():
        if isinstance(sum_value, tuple):
            averages[key] = tuple(x / num_items for x in sum_value)
        else:
            averages[key] = sum_value / num_items
    
    return averages



folder_root = Path('results')
lr = 1e-07
for method in ['raw', 'ft', 'ga', 'gaft', 'salun', 'ssd']:
    files = folder_root.glob(f"*_{method}.txt")
        
    data_list = []
    for file in sorted(list(files)):
        results = parse_file(file)
    
        # Find the first zero or the lowest fgt_acc1
        first_zero_index = None
        min_value = float('inf')
        min_index = None
        
        for i, acc in enumerate(results['fgt_acc1']):
            if acc == 0 and first_zero_index is None:
                first_zero_index = i
                break
            if acc < min_value:
                min_value = acc
                min_index = i
        
        # Use first zero if available, otherwise use the index of the minimum value
        target_index = first_zero_index if first_zero_index is not None else min_index
            
        # Output the data at the found index
        output = {key: values[target_index] for key, values in results.items()}
        data_list.append(output)
        # print(output)
    
    averages = calculate_averages(data_list)
    # print(method, averages)

    s = f'{method}: '
    for i in averages:
        if isinstance(averages[i], (int, float)):
            s += f"{i}: {averages[i]*100:.2f}%, "
        else:
            s += f"{i}: {averages[i][0]:.2f} ± {averages[i][1]:.2f} "
    print(s)
