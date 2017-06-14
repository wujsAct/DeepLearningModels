#@time: 2017/6/13

dir_path="data/kbp/LDC2017EDL/data/2014/"
dims="100"

:<<!
generate coreference results and entMentsTags.p
!
python utils/getCoref.py --dir_path ${dir_path} --data_tag ${data_tag}