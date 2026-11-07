"""Detector-only loaders matching SAMBAKE (Meta sam3 f1647e8). GPU imports stay server-side."""
import os


def load_text_model(name, root):
    from sam3.model_builder import build_sam3_image_model

    checkpoint = os.path.join(root, "sam3/sam3.pt" if name == "sam3" else
                              "sam3.1/sam3.1_multiplex_fp16.safetensors")
    if name == "sam3.1_text":
        from safetensors.torch import load_file
        import sam3.model_builder as mb
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        import pkg_resources

        # Construct Meta's multiplex detector, without its unused video tracker.
        # Components/arguments follow sam3.model_builder at f1647e8.
        tri = mb._create_multiplex_tri_backbone(
            compile_mode=None, use_fa3=False, use_rope_real=False
        )
        text = mb._create_text_encoder(
            pkg_resources.resource_filename(
                "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
        model = Sam3MultiplexDetector(
            num_feature_levels=1,
            backbone=mb.SAM3VLBackboneTri(scalp=0, visual=tri, text=text),
            transformer=mb._create_sam3_transformer(use_fa3=False),
            segmentation_head=mb._create_segmentation_head(use_fa3=False),
            semantic_segmentation_head=None,
            input_geometry_encoder=mb._create_geometry_encoder(),
            use_early_fusion=True,
            use_dot_prod_scoring=True,
            dot_prod_scoring=mb._create_dot_product_scoring(),
            supervise_joint_box_scores=True,
            is_multiplex=True,
        )
        ck = load_file(checkpoint)
        detector = {
            k.removeprefix("detector."): v
            for k, v in ck.items()
            if k.startswith("detector.")
        }
        missing, unexpected = model.load_state_dict(detector, strict=False)
        # Official loader also permits the unused CLIP projection and computed RoPE buffers.
        if unexpected or any(
            not (k.endswith("freqs_cis") or k.endswith("text_projection"))
            for k in missing
        ):
            raise RuntimeError((missing, unexpected))
        model = model.eval().cuda()
        del ck, detector
    else:
        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_inst_interactivity=False,
        )
    return model
