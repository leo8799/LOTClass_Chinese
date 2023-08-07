from src.trainer import LOTClassTrainer
from config.configs_interface import configs as args
trainer = LOTClassTrainer(args)
trainer.category_vocabulary(top_pred_num=args.train_args.top_pred_num,
                            category_vocab_size=args.train_args.category_vocab_size)
trainer.mcp(args.train_args.top_pred_num, args.train_args.match_threshold, args.train_args.SELF_TRAIN_EPOCH, "mcp_model.pt")
trainer.self_train(epochs=args.train_args.SELF_TRAIN_EPOCH, loader_name=args.data.final_model)

trainer.write_results(loader_name=args.data.final_model, out_file=args.data.out_file)



