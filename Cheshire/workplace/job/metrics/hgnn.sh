#!/bin/bash

models=("iJN1463" "iMM904" "iSB619" "iE2348C_1286" "iIS312" "iAF1260b" "iS_1188" "iCN718" "iAF692" "iAM_Pb448" "iJB785" "iAF987" "iJO1366" "iML1515" "iIT341" "iJN678" "iECB_1328" "iJR904" "e_coli_core" "iAB_RBC_283" "iAF1260" "iAM_Pc455" "iAM_Pf480" "iAM_Pk459" "iAM_Pv461" "iAPECO1_1312" "iAT_PLT_636" "iB21_1397" "iBWG_1329" "ic_1306" "iCHOv1" "iCHOv1_DG44" "iCN900" "iEC042_1314" "iEC1344_C" "iEC1349_Crooks" "iEC1356_Bl21DE3" "iEC1364_W" "iEC1368_DH5a" "iEC1372_W3110" "iEC55989_1330" "iECABU_c1320" "iECBD_1354" "iECD_1391" "iEcDH1_1363" "iECDH1ME8569_1439" "iECDH10B_1368" "iEcE24377_1341" "iECED1_1282" "iECH74115_1262" "iEcHS_1320" "iECIAI1_1343" "iECIAI39_1322" "iECNA114_1301" "iECO26_1355" "iECO103_1326" "iECO111_1330" "iECOK1_1307" "iEcolC_1368" "iECP_1309" "iECs_1301" "iECS88_1305" "iECSE_1348" "iECSF_1327" "iEcSMS35_1347" "iECSP_1301" "iECUMN_1333" "iECW_1372" "iEK1008" "iEKO11_1354" "iETEC_1333" "iG2583_1286" "iHN637" "iIS312_Amastigote" "iIS312_Epimastigote" "iIS312_Trypomastigote" "iJN746" "iLB1027_lipid" "iLF82_1304" "iLJ478" "iMM1415" "iND750" "iNF517" "iNJ661" "iNRG857_1313" "iPC815" "iRC1080" "iSbBS512_1146" "iSBO_1134" "iSDY_1059" "iSF_1195" "iSFV_1184" "iSFxv_1172" "iSSON_1240" "iSynCJ816" "iUMN146_1321" "iUMNK88_1353" "iUTI89_1310" "iWFL_1372" "iY75_1357" "iYL1228" "iYO844" "iYS854" "iYS1720" "iZ_1308" "RECON1" "Recon3D" "STM_v1_0")
# models=("iSFxv_1172" "iEcSMS35_1347")
# models=("iAF692" "iSB619" "iMM904" "iAF1260b")
# models=("iE2348C_1286")
# models=("iMM1415" "iECED1_1282" "iEC55989_1330" "iB21_1397" "iECB_1328" "iJO1366" "iCN718")

for model in "${models[@]}"
do
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${model}_hgnn
#SBATCH --output=/nas/longleaf/home/cyixiao/Project/Cheshire/workplace/slurm/%x/%j/out.txt
#SBATCH --error=/nas/longleaf/home/cyixiao/Project/Cheshire/workplace/slurm/%x/%j/error.txt
#SBATCH --partition=a100-gpu,volta-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

cd /nas/longleaf/home/cyixiao
source test_env/bin/activate
cd Project/Cheshire

python main.py --name "$model" --repeat 20 --model "HGNN" --lr 0.001 --emb_dim 256 --max_epoch 200

EOT
done