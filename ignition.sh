for mid in wicket wicket-clni wicket-clnifarsec wicket-clnifarsecsq wicket-clnifarsectwo wicket-farsec wicket-farsecsq wicket-farsectwo \
ambari ambari-clni ambari-clnifarsec ambari-clnifarsecsq ambari-clnifarsectwo ambari-farsec ambari-farsecsq ambari-farsectwo \
camel camel-clni camel-clnifarsec camel-clnifarsecsq camel-clnifarsectwo camel-farsec camel-farsecsq camel-farsectwo \
derby derby-clni derby-clnifarsec derby-clnifarsecsq derby-clnifarsectwo derby-farsec derby-farsecsq derby-farsectwo \
chromium chromium-clni chromium-clnifarsec chromium-clnifarsecsq chromium-clnifarsectwo chromium-farsec chromium-farsecsq chromium-farsectwo
do
	export mid
	echo $mid
	sbatch epsilon.batch
done
