from src.trainer import LOTClassTrainer
from config.configs_interface import configs as args
import os
from src.logers import LOGS
LOGS.init(os.path.join(args.project.PROJECT_DIR, "{}/{}".format(args.log.log_dir, args.log.log_file_name)))

trainer = LOTClassTrainer(args)
trainer.category_vocabulary(top_pred_num=args.train_args.top_pred_num,
                            category_vocab_size=args.train_args.category_vocab_size)
trainer.mcp(args.train_args.top_pred_num, args.train_args.match_threshold, args.train_args.MCP_EPOCH)
trainer.self_train(epochs=args.train_args.SELF_TRAIN_EPOCH, loader_name=args.data.final_model)
if args.data.TEST_CORPUS is not None:
    trainer.write_results(loader_name=args.data.final_model, out_file=args.data.out_file)



