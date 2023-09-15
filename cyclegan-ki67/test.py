    # ----------------------------------------------------------------------
    # 转换tfpb模型为h5
    # ----------------------------------------------------------------------

    # for k in range(10, 11):
    #     start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     )
    #     print(start_path)
    #     model = tf.keras.models.load_model(start_path)
    #     model.save(model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5"% (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     ))
        #model.evaluate(valGene, steps=n_val//bs_v)
        # kappa = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=True)
        # kappa.update_state(y_true , y_pred)