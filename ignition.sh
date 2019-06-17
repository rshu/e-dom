for mid in wicket ambari camel derby chromium
do
	export mid
	echo $mid
	sbatch epsilon.batch
done
