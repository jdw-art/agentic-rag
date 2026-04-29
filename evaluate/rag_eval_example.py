
"""
劳动法领域RAG评估系统完整运行脚本
功能：加载法律文档→构建向量库→生成问答数据→RAGAS评估→结果可视化
说明：运行前需修改配置类中的路径、API Key等参数，确保依赖库已安装
依赖库安装：pip install langchain chromadb huggingface-hub datasets ragas seaborn matplotlib pandas
"""

# 1. 导入所有依赖库

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# LangChain相关
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 数据集评估相关
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, context_recall, context_precision



# 2. 参数配置

class Config:
    # 嵌入模型本地路径（需确保模型已下载到该目录）
    EMBED_MODEL_PATH = r"C:\hugging_face\GTE"

    # 数据与向量库路径（相对路径，不存在会自动创建）
    DATA_DIR = "data"  # 存放JSON格式法律文档的目录
    VECTOR_DB_DIR = "chroma_dbv2"  # 向量库持久化目录
    COLLECTION_NAME = "chinese_labor_laws"  # 向量库集合名
    TOP_K = 3  # 检索时返回的Top K条文档

    # DeepSeek LLM配置（替换为你自己的API Key）
    DEEPSEEK_API_KEY = "xxxxxxx"  # 你的API Key
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"  # 模型名称



# 3. 工具函数定义（加载文档、初始化模型/向量库）
def load_json_documents(data_dir: str) -> List[Dict]:
    """
    加载指定目录下的所有JSON格式法律文档，提取内容与元数据
    :param data_dir: JSON文档所在目录
    :return: 包含page_content和metadata的文档列表
    """
    # 查找目录下所有.json文件
    json_files = list(Path(data_dir).glob("*.json"))
    print(f"📄 找到JSON文件数量：{len(json_files)}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"在{data_dir}目录下未找到JSON文件，请检查路径")

    docs = []
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 假设JSON结构为：[{"法律名称 条款号": "条款内容"}, ...]
                for item in data:
                    for title, content in item.items():
                        # 提取元数据（便于后续追溯来源）
                        metadata = {
                            "source_file": file.name,
                            "full_title": title,
                            "law_name": title.split(" ")[0] if " " in title else title,
                            "article": title.split(" ")[1] if " " in title else "未知条款"
                        }
                        docs.append({"page_content": content, "metadata": metadata})
        except Exception as e:
            print(f"⚠️ 加载文件{file.name}时出错：{str(e)}，跳过该文件")

    print(f"✅ 成功加载有效文档数量：{len(docs)}")
    if len(docs) == 0:
        raise ValueError("未加载到任何有效文档，请检查JSON文件格式")
    return docs


def init_models() -> (HuggingFaceEmbeddings, ChatOpenAI):
    """
    初始化嵌入模型（本地HuggingFace模型）和LLM（DeepSeek）
    :return: 嵌入模型实例、LLM实例
    """
    # 初始化嵌入模型
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL_PATH,
            model_kwargs={"trust_remote_code": True}  # 加载自定义模型需开启
        )
    except Exception as e:
        raise RuntimeError(f"初始化嵌入模型失败：{str(e)}，请检查模型路径是否正确")

    # 初始化DeepSeek LLM（伪装为ChatOpenAI调用）
    try:
        llm = ChatOpenAI(
            model_name=Config.DEEPSEEK_MODEL,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_API_BASE,
            temperature=0.3  # 低温度确保回答严谨（法律场景）
        )
    except Exception as e:
        raise RuntimeError(f"初始化LLM失败：{str(e)}，请检查API Key和Base URL")

    print("✅ 嵌入模型与LLM均初始化完成")
    return embedding_model, llm


def init_vectorstore(docs: List[Dict], embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    初始化向量库（存在则加载，不存在则新建并持久化）
    :param docs: 加载好的文档列表（含page_content和metadata）
    :param embedding_model: 嵌入模型实例
    :return: Chroma向量库实例
    """
    vectorstore_path = Path(Config.VECTOR_DB_DIR)

    # 新建向量库（路径不存在时）
    if not vectorstore_path.exists():
        print(f"📦 未找到向量库，正在{Config.VECTOR_DB_DIR}目录下新建...")
        texts = [doc["page_content"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]

        try:
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas,
                persist_directory=Config.VECTOR_DB_DIR,
                collection_name=Config.COLLECTION_NAME
            )
            vectorstore.persist()  # 持久化向量库（避免下次重新构建）
            print("✅ 新向量库创建并持久化完成")
        except Exception as e:
            raise RuntimeError(f"创建向量库失败：{str(e)}")

    # 加载已有向量库
    else:
        print(f"📂 找到现有向量库，正在从{Config.VECTOR_DB_DIR}加载...")
        try:
            vectorstore = Chroma(
                embedding_function=embedding_model,
                persist_directory=Config.VECTOR_DB_DIR,
                collection_name=Config.COLLECTION_NAME
            )
            print("✅ 现有向量库加载完成")
        except Exception as e:
            raise RuntimeError(f"加载向量库失败：{str(e)}，可删除{Config.VECTOR_DB_DIR}后重试")

    # 打印向量库文档数量（验证加载结果）
    try:
        doc_count = vectorstore._collection.count()
        print(f"📊 向量库中当前存储的文档总数：{doc_count}")
    except Exception as e:
        print(f"⚠️ 无法获取向量库文档数量（可能是Chroma版本差异）：{str(e)}")

    return vectorstore



# 4. 核心流程：RAG问答生成 + RAGAS评估
def main():
    # 步骤1：初始化基础组件（文档→模型→向量库）
    print("\n" + "=" * 50)
    print("步骤1：初始化基础组件")
    print("=" * 50)

    # 加载JSON文档
    docs = load_json_documents(Config.DATA_DIR)

    # 初始化嵌入模型与LLM
    embedding_model, llm = init_models()

    # 初始化向量库
    vectorstore = init_vectorstore(docs, embedding_model)

    # 构建检索器（Top K=3）
    retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K})
    print("✅ 检索器构建完成（Top K={Config.TOP_K}）")

    # 步骤2：定义Prompt模板
    print("\n" + "=" * 50)
    print("步骤2：定义Prompt模板")
    print("=" * 50)

    QA_TEMPLATE = (
        "你是专业的劳动法助手，必须严格根据以下提供的法律条文回答用户问题，不添加任何未提及的信息：\n"
        "-------------------\n"
        "相关法律条文：\n{context}\n"
        "-------------------\n"
        "用户问题：\n{question}\n"
        "要求：1. 仅用条文内容回答，不扩展解释；2. 若条文未覆盖问题，直接回复'暂无相关法律条文支持'\n"
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_TEMPLATE
    )
    print("✅ 法律场景Prompt模板定义完成")


    # 步骤3：构建RAG链并生成问答数据
    print("\n" + "=" * 50)
    print("步骤3：生成RAG问答数据（共{len(questions)}个问题）")
    print("=" * 50)

    # 劳动法高频问题列表（由大模型生成）
    questions = [
        "试用期最长可以多长时间？",
        "公司是否可以只约定试用期不签正式合同？",
        "在什么情况下劳动者可以解除劳动合同？",
        "用人单位拖欠工资怎么办？",
        "工作期间女职工怀孕了，公司是否可以辞退她？",
        "劳动合同必须具备哪些基本条款？",
        "什么时候必须签署书面劳动合同？",
        "劳动者未签订书面劳动合同可以获得什么补偿？",
        "用人单位什么时候必须支付双倍工资？",
        "什么是无固定期限劳动合同？",
        "如何判断劳动合同是否无效？",
        "非全日制用工的工资结算周期最长是多少？",
        "单位可以向劳动者收取押金或扣证件吗？",
        "公司合并后原劳动合同是否继续有效？",
        "派遣工是否享有同工同酬的权利？",
        "劳务派遣适用于哪些岗位？",
        "在试用期内辞职需要提前几天通知？",
        "公司没有明确工资标准，应按什么支付？",
        "劳动者患病后不能继续工作，公司可以辞退吗？",
        "什么情况下公司需要向员工支付经济补偿？"
    ]

    # 人工标准答案（由大模型生成）
    ground_truths = [
        [
            "试用期不得超过六个月，具体期限根据劳动合同期限确定：劳动合同期限三个月以上不满一年的，试用期不得超过一个月；劳动合同期限一年以上不满三年的，试用期不得超过二个月；三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。"],
        ["不可以。劳动合同仅约定试用期的，试用期不成立，该期限视为劳动合同期限。"],
        [
            "劳动者可以解除劳动合同的情形包括：（1）用人单位未按照劳动合同约定提供劳动保护或者劳动条件的；（2）用人单位未及时足额支付劳动报酬的；（3）用人单位未依法为劳动者缴纳社会保险费的；（4）用人单位的规章制度违反法律、法规的规定，损害劳动者权益的；（5）用人单位以欺诈、胁迫的手段或者乘人之危，使劳动者在违背真实意思的情况下订立或者变更劳动合同的；（6）法律、行政法规规定劳动者可以解除劳动合同的其他情形。"],
        [
            "用人单位拖欠或者未足额支付劳动报酬的，劳动者可以依法向当地人民法院申请支付令，人民法院应当依法发出支付令。同时，劳动者也可以解除劳动合同，并要求用人单位支付经济补偿。"],
        [
            "不可以。女职工在孕期、产期、哺乳期的，用人单位不得依照《劳动合同法》第四十条（非过失性辞退）、第四十一条（经济性裁员）的规定解除劳动合同。"],
        [
            "劳动合同应当具备以下条款：（1）用人单位的名称、住所和法定代表人或者主要负责人；（2）劳动者的姓名、住址和居民身份证或者其他有效身份证件号码；（3）劳动合同期限；（4）工作内容和工作地点；（5）工作时间和休息休假；（6）劳动报酬；（7）社会保险；（8）劳动保护、劳动条件和职业危害防护；（9）法律、法规规定应当纳入劳动合同的其他事项。"],
        ["用人单位自用工之日起即与劳动者建立劳动关系，应当自用工之日起一个月内订立书面劳动合同。"],
        ["用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。"],
        [
            "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当依照《劳动合同法》第八十二条的规定向劳动者每月支付两倍的工资，并与劳动者补订书面劳动合同；劳动者不与用人单位订立书面劳动合同的，用人单位应当书面通知劳动者终止劳动关系，并依照《劳动合同法》第四十七条的规定支付经济补偿。"],
        [
            "无固定期限劳动合同，是指用人单位与劳动者约定无确定终止时间的劳动合同。用人单位与劳动者协商一致，可以订立无固定期限劳动合同。有下列情形之一，劳动者提出或者同意续订、订立劳动合同的，除劳动者提出订立固定期限劳动合同外，应当订立无固定期限劳动合同：（1）劳动者在该用人单位连续工作满十年的；（2）用人单位初次实行劳动合同制度或者国有企业改制重新订立劳动合同时，劳动者在该用人单位连续工作满十年且距法定退休年龄不足十年的；（3）连续订立二次固定期限劳动合同，且劳动者没有《劳动合同法》第三十九条和第四十条第一项、第二项规定的情形，续订劳动合同的。"],
        [
            "下列劳动合同无效或者部分无效：（1）以欺诈、胁迫的手段或者乘人之危，使对方在违背真实意思的情况下订立或者变更劳动合同的；（2）用人单位免除自己的法定责任、排除劳动者权利的；（3）违反法律、行政法规强制性规定的。对劳动合同的无效或者部分无效有争议的，由劳动争议仲裁机构或者人民法院确认。"],
        ["非全日制用工劳动报酬结算支付周期最长不得超过十五日。"],
        [
            "用人单位招用劳动者，不得扣押劳动者的居民身份证和其他证件，不得要求劳动者提供担保或者以其他名义向劳动者收取财物。"],
        ["用人单位发生合并或者分立等情况，原劳动合同继续有效，劳动合同由承继其权利和义务的用人单位继续履行。"],
        [
            "被派遣劳动者享有与用工单位的劳动者同工同酬的权利。用工单位应当按照同工同酬原则，对被派遣劳动者与本单位同类岗位的劳动者实行相同的劳动报酬分配办法。用工单位无同类岗位劳动者的，参照用工单位所在地相同或者相近岗位劳动者的劳动报酬确定。"],
        [
            "劳务派遣一般在临时性、辅助性或者替代性的工作岗位上实施。临时性工作岗位是指存续时间不超过六个月的岗位；辅助性工作岗位是指为主营业务岗位提供服务的非主营业务岗位；替代性工作岗位是指用工单位的劳动者因脱产学习、休假等原因无法工作的一定期间内，可以由其他劳动者替代工作的岗位。"],
        ["劳动者在试用期内提前三日通知用人单位，可以解除劳动合同。"],
        [
            "用人单位未在用工的同时订立书面劳动合同，与劳动者约定的劳动报酬不明确的，新招用的劳动者的劳动报酬按照集体合同规定的标准执行；没有集体合同或者集体合同未规定的，实行同工同酬。"],
        [
            "劳动者患病或者非因工负伤，在规定的医疗期满后不能从事原工作，也不能从事由用人单位另行安排的工作的，用人单位提前三十日以书面形式通知劳动者本人或者额外支付劳动者一个月工资后，可以解除劳动合同，但应当向劳动者支付经济补偿。"],
        [
            "有下列情形之一的，用人单位应当向劳动者支付经济补偿：（1）劳动者依照《劳动合同法》第三十八条规定解除劳动合同的；（2）用人单位依照《劳动合同法》第三十六条规定向劳动者提出解除劳动合同并与劳动者协商一致解除劳动合同的；（3）用人单位依照《劳动合同法》第四十条规定解除劳动合同的；（4）用人单位依照《劳动合同法》第四十一条第一款规定解除劳动合同的；（5）除用人单位维持或者提高劳动合同约定条件续订劳动合同，劳动者不同意续订的情形外，依照《劳动合同法》第四十四条第一项规定终止固定期限劳动合同的；（6）依照《劳动合同法》第四十四条第四项、第五项规定终止劳动合同的；（7）法律、行政法规规定的其他情形。"]
    ]

    # 存储RAG生成的回答和检索上下文
    answers = []
    contexts = []

    # 构建RAG链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # 适合短文档（法律条文）的拼接方式
        return_source_documents=True,  # 返回检索到的源文档（用于上下文记录）
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 批量生成问答结果
    for idx, query in enumerate(questions, 1):
        print(f"🔍 正在处理第{idx}/{len(questions)}个问题：{query}")
        try:
            result = qa_chain(query)
            answers.append(result["result"])  # 存储生成的回答
            # 存储检索到的上下文（提取page_content）
            context_docs = result["source_documents"]
            contexts.append([doc.page_content for doc in context_docs])
        except Exception as e:
            print(f"⚠️ 处理问题'{query}'时出错：{str(e)}，跳过该问题")
            answers.append("生成失败")
            contexts.append(["检索失败"])

    print("✅ RAG问答数据生成完成")


    # 步骤4：构建RAGAS评估数据集
    print("\n" + "=" * 50)
    print("步骤4：构建RAGAS评估数据集")
    print("=" * 50)

    # 确保数据长度一致（过滤生成失败的样本）
    valid_indices = [i for i, ans in enumerate(answers) if ans != "生成失败"]
    if len(valid_indices) < len(questions):
        print(f"⚠️ 过滤掉{len(questions) - len(valid_indices)}个生成失败的样本，有效样本数：{len(valid_indices)}")
        questions = [questions[i] for i in valid_indices]
        answers = [answers[i] for i in valid_indices]
        contexts = [contexts[i] for i in valid_indices]
        ground_truths = [ground_truths[i] for i in valid_indices]

    # 构建RAGAS要求的数据集格式（必须包含以下4个字段）
    data = {
        "user_input": questions,  # 用户问题
        "response": answers,  # RAG生成的回答
        "retrieved_contexts": contexts,  # 检索到的上下文
        "reference": ground_truths  # 标准答案
    }

    # 转换为datasets.Dataset格式
    dataset = Dataset.from_dict(data)
    print(f"✅ 评估数据集构建完成，样本结构：")
    print(dataset[0])  # 打印第一个样本示例

    # --------------------------
    # 步骤5：RAGAS评估（计算3个核心指标）
    # --------------------------
    print("\n" + "=" * 50)
    print("步骤5：运行RAGAS评估")
    print("=" * 50)

    # 包装LLM为RAGAS可调用格式
    evaluator_llm = LangchainLLMWrapper(llm)

    # 执行评估（指定需要计算的指标）
    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness],  # 3个核心指标
            llm=evaluator_llm,
            verbose=True  # 打印评估过程日志
        )
    except Exception as e:
        raise RuntimeError(f"RAGAS评估失败：{str(e)}")

    # 转换评估结果为DataFrame（便于查看和后续分析）
    result_df = evaluation_result.to_pandas()
    # 设置DataFrame显示选项（完整显示长文本）
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)


    print("\n" + "=" * 50)
    print("各样本详细评估结果")
    print("=" * 50)
    print(result_df[["question", "answer", "context_precision", "context_recall", "faithfulness"]])


    # 步骤6：评估结果可视化（箱线图）
    print("\n" + "=" * 50)
    print("步骤6：生成评估指标箱线图")
    print("=" * 50)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制箱线图
    metric_cols = ["context_precision", "context_recall", "faithfulness"]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=result_df[metric_cols], palette="Set2")

    # 设置图表标题和标签
    plt.title("劳动法RAG系统评估指标分布箱线图", fontsize=14, fontweight="bold")
    plt.ylabel("指标得分（0-1）", fontsize=12)
    plt.xlabel("评估指标", fontsize=12)
    plt.ylim(0, 1.1)  # 限定y轴范围（0-1.1），便于观察分布
    plt.grid(axis="y", alpha=0.3)  # 添加y轴网格线

    # 保存图片
    plt.savefig("rag_evaluation_boxplot.png", dpi=300, bbox_inches="tight")
    print("✅ 箱线图已保存为：rag_evaluation_boxplot.png")

    # 显示图片
    plt.show()

    # 保存详细结果到CSV
    result_df.to_csv("rag_evaluation_detailed_result.csv", index=False, encoding="utf-8-sig")
    print("✅ 详细评估结果已保存为：rag_evaluation_detailed_result.csv")


if __name__ == "__main__":
    try:
        print("🎉 劳动法RAG系统评估脚本开始运行")
        main()
        print("\n🎉 所有流程执行完成！")
    except Exception as e:
        print(f"\n❌ 脚本运行失败：{str(e)}")