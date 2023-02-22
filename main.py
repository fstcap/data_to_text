import argparse

parser = argparse.ArgumentParser(description='select run way')
parser.add_argument("--model", type=str, default="train")  # train.
parser.add_argument("--memory_limit", type=int, default=22)  # 限制显存大小.
parser.add_argument("--prename", type=str, default="gpt2-large")  # gpt2, gpt2-large.
parser.add_argument("--epochs", type=int, default=10)  # 正整数.
parser.add_argument("--batch_size", type=int, default=32)  # 正整数.
parser.add_argument("--input_max_length", type=int, default=512)  # 正整数.
parser.add_argument("--output_max_length", type=int, default=128)  # 正整数.
parser.add_argument("--lr_max_value", type=float, default=0.001)  # 正数.
parser.add_argument("--datasets", type=str, default='records_2023-02-17_17-40-47.pkl')  #

args = parser.parse_args()

if __name__ == "__main__":
    if args.model == "train":
        from model import GPT2ConditionalGeneration

        gpt2_conditional_generation = GPT2ConditionalGeneration(
            memory_limit=args.memory_limit,
            pre_m_name_or_path=args.prename,
            epochs=args.epochs,
            batch_size=args.batch_size,
            input_max_length=args.input_max_length,
            output_max_length=args.output_max_length,
            lr_max_value=args.lr_max_value,
            datasets_name=args.datasets)
        gpt2_conditional_generation()
