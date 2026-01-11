# 第一批
# nohup bash run.sh origin 0 8010 cora gt True 5 './exps/cora_max_dist_5' > cora_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 8011 cora gt True 7 './exps/cora_max_dist_7' 8011 > cora_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 8012 citeseer gt True 5 './exps/citeseer_max_dist_5' > citeseer_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 8013 citeseer gt True 7 './exps/citeseer_max_dist_7' > citeseer_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 0 8010 pubmed gt True 5 './exps/pubmed_max_dist_5' > pubmed_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 8011 pubmed gt True 7 './exps/pubmed_max_dist_7' > pubmed_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 8012 ogbn-arxiv gt True 2 './exps/ogbn_arxiv_max_dist_5' > ogbn_arxiv_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 8013 ogbn-arxiv gt True 4 './exps/ogbn_arxiv_max_dist_7' > ogbn_arxiv_max_dist_7.log 2>&1 &

# nohup bash run.sh metis 1 8010 cora gt True 7 './exps/cora_max_dist_7' > metis_cora_max_dist_7_enc.log 2>&1 &
# nohup bash run.sh metis 2 8011 cora gt False 7 './exps/cora_max_dist_7' > metis_cora_max_dist_7_.log 2>&1 &
# nohup bash run.sh metis 3 8012 citeseer gt True 7 './exps/citeseer_max_dist_7' > metis_citeseer_max_dist_7_enc.log 2>&1 &
# nohup bash run.sh metis 4 8013 citeseer gt False 7 './exps/citeseer_max_dist_7' > metis_citeseer_max_dist_7_.log 2>&1 &

# nohup bash run.sh origin 0 8010 cora gt False 5 './exps/cora_max_dist_5_unenc' > cora_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 1 8011 cora gt False 7 './exps/cora_max_dist_7_unenc' > cora_max_dist_7.log 2>&1 &
# nohup bash run.sh origin 2 8012 citeseer gt False 5 './exps/citeseer_max_dist_5_unenc' > citeseer_max_dist_5.log 2>&1 &
# nohup bash run.sh origin 3 8013 citeseer gt False 7 './exps/citeseer_max_dist_7_unenc' > citeseer_max_dist_7.log 2>&1 &