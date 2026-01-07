# 第一批
# nohup bash run.sh origin 0 cora True 5 './exps/cora_max_dist_5' 8010 > cora_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 cora True 7 './exps/cora_max_dist_7' 8011 > cora_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 citeseer True 5 './exps/citeseer_max_dist_5' 8012 > citeseer_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 citeseer True 7 './exps/citeseer_max_dist_7' 8013 > citeseer_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 0 pubmed True 5 './exps/pubmed_max_dist_5' 8012 > pubmed_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 pubmed True 7 './exps/pubmed_max_dist_7' 8013 > pubmed_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 ogbn-arxiv True 2 './exps/ogbn_arxiv_max_dist_5' 8012 > ogbn_arxiv_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 ogbn-arxiv True 4 './exps/ogbn_arxiv_max_dist_7' 8013 > ogbn_arxiv_max_dist_7.log 2>&1 &

# nohup bash run.sh metis 1 cora True 7 './exps/cora_max_dist_7' 8010 > metis_cora_max_dist_7_enc.log 2>&1 &
# nohup bash run.sh metis 2 cora False 7 './exps/cora_max_dist_7' 8011 > metis_cora_max_dist_7_.log 2>&1 &
# nohup bash run.sh metis 3 citeseer True 7 './exps/citeseer_max_dist_7' 8010 > metis_citeseer_max_dist_7_enc.log 2>&1 &
# nohup bash run.sh metis 4 citeseer False 7 './exps/citeseer_max_dist_7' 8011 > metis_citeseer_max_dist_7_.log 2>&1 &

# nohup bash run.sh origin 0 cora False 5 './exps/cora_max_dist_5_unenc' 8010 > cora_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 cora False 7 './exps/cora_max_dist_7_unenc' 8011 > cora_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 citeseer False 5 './exps/citeseer_max_dist_5_unenc' 8012 > citeseer_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 citeseer False 7 './exps/citeseer_max_dist_7_unenc' 8013 > citeseer_max_dist_7.log 2>&1 &