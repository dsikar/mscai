# MScAI

City University of London MSc Artificial Intellingence High Performance Computing Cluster (Hyperion) Scripts

## Connecting to the Hyperion cluster

* [Connect with Windows OS from inside City network - login required] (https://cityuni.service-now.com/sp?sys_kb_id=a8d1608c1b214950f82d9828b04bcb8a&id=kb_article_view&sysparm_rank=1&sysparm_tsqueryId=0b2561a61b5e41104e86b886d34bcbe8)

* Connect with from outside network
1. [Setup Pulse VPN - login required] (https://cityuni.service-now.com/sp?id=kb_article_view&sys_kb_id=e13aac9edba784d0fa2415784b9619e8)
2. Open terminals (putty on Windows OS) like connecting from inside City network
NB Pulse VPN can be setup on Linux, then two terminal connections are required, as per inside the network.


## Running jobs

See Chris Marshall's [guides] (https://cityuni.service-now.com/sp?id=kb_article_view&sysparm_article=KB0012621&sys_kb_id=7bf82adc1b210d50f82d9828b04bcb9b&spa=1)

Examples:
```
<gridware{+}> [aczd097@login1 [hyperion] Lab1]$ sbatch /users/aczd097/localscratch/mscai/Lab1/runModel.sh
<gridware{+}> [aczd097@login1 [hyperion] Lab1]$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            173493    gengpu c_bt_Bla  adbb281 PD       0:00      1 (QOSMaxGRESPerUser)
            173492    gengpu c_bt_SCT  adbb281 PD       0:00      1 (QOSMaxGRESPerUser)
            170447    gengpu    LU_34  aczd082  R   23:25:50      1 gpu02
            173490    gengpu c_bt_5K_  adbb281  R    5:12:15      1 gpu02
            166274     nodes    trip1  sbrs173  R 2-18:43:27      1 node023
            166277     nodes    trip2  sbrs173  R 2-18:25:02      1 node034
            166314     nodes    trip3  sbrs173  R 2-08:03:03      1 node004
            173506     nodes    inc60  acjw990  R      49:38      4 node[011-014]
            173505     nodes    inc60  acjw990  R      52:20      4 node[066-069]
            173504     nodes runfluen  aczk895  R    1:56:04      3 node[001-003]
            173497     nodes 50_125M1  sbrn822  R    2:21:55      1 node033
            173496     nodes  fluent2  xbkw312  R    3:09:11      6 node[005-010]
            173495     nodes runfluen  aczk895  R    4:11:20      3 node[027-029]
            173478     nodes 30_125M1  sbrn822  R 1-01:25:26      1 node062
            173477     nodes 51_125M1  sbrn822  R 1-01:28:24      1 node061
            173476     nodes 52_125M1  sbrn822  R 1-01:28:30      1 node060
            173474     nodes 50_125M1  sbrn822  R 1-01:30:32      1 node032
            170468     nodes my_fluen  sbrn785  R 1-06:48:25      2 node[024-025]
            168416     nodes A6_ratio  adbf954  R 2-02:07:07      1 node057
            168417     nodes D6_ratio  adbf954  R 2-02:07:07      1 node059
            168418     nodes E6_ratio  adbf954  R 2-02:07:07      1 node065
            168419     nodes E7_ratio  adbf954  R 2-02:07:07      1 node070
            168420     nodes E8_ratio  adbf954  R 2-02:07:07      1 node071
            168421     nodes B6_ratio  adbf954  R 2-02:07:07      1 node072
```



