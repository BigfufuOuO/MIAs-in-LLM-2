from data.enron import EnronDataset
from data.echr import EchrDataset
from models.ft_clm import FinetunedCasualLM as HFModels
from attacks.MIA.member_inference import MemberInferenceAttack, MIAMetric
from metrics import JailbreakRate

data = EchrDataset(data_path="data/echr")
# data = EnronDataset(data_path="data/enron")
# Fill api_key
llm = HFModels(model_path="openai-community/gpt2")
ref_model = HFModels(model_path="openai-community/gpt2",)
attack = MemberInferenceAttack(metric=MIAMetric.MIN_K_PROB, ref_model=ref_model, n_neighbor=15)
# attack = MemberInferenceAttack(metric=MIAMetric.PPL)
results = attack.execute(llm, data.train_set(), data.test_set(), cache_file='cache.pth')
rate = attack.evaluate(results)
print("rate:", rate)
