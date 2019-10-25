for file in ./*
do
convert -border 2x2 -bordercolor "#000000" "$file" "$file"
done
