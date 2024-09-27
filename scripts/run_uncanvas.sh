# Perform model update after calculating the gradients
cd src

unlearn_tasks=(
  "style"
  "class"
)


for unlearn_task in "${unlearn_tasks[@]}"; do
  echo "Unlearning $unlearn_task on UnlearnCanvas:"
  
  python -m clip.a6_uncanvas \
      --uncanvas='../data/UnlearnCanvas' \
      --unlearn_task $unlearn_task \
  
done

