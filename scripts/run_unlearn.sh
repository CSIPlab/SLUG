# Perform model update after calculating the gradients
cd src

celeb_names=(
  "Elon_Musk"
  "Mark_Zuckerberg"
  "Jeff_Bezos"
  "Taylor_Swift"
  "Kim_Kardashian"
  "Kanye_West"
  "Barack_Obama"
  "Bruce_Lee"
  "Fan_Bingbing"
  "Lady_Gaga"
)
  # Iterate over the array
  # for mask_layer in "${layers[@]}"; do
  #   echo "Processing layer: $mask_layer"
  #   # Here you can add the code to process each layer

for celeb_name in "${celeb_names[@]}"; 
do    
        echo "Processing name: $celeb_name"
        python -m clip.a6_binary_search_auto --celeb_name $celeb_name

done

# for celeb_name in Mark_Zuckerberg 
# python -m clip.a6_binary_search_auto --celeb_name Mark_Zuckerberg

