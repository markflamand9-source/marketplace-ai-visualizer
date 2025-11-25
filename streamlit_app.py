# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own photos and Market & Place products. "
        "You can either upload a room for AI styling, or generate a store shelf concept "
        "using the Market & Place shelf photo."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM CONCEPT IMAGE (image-to-image) ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (bathroom, bedroom, etc.):",
            type=["jpg", "jpeg", "png"],
            key="room_upload",
        )

        room_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. luxury towels, navy striped bath towels, white queen quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if st.button("Generate room concept image"):
            if uploaded_room is None:
                st.error("Please upload a room photo first.")
            else:
                with st.spinner("Generating AI room concept…"):
                    try:
                        img_bytes = generate_room_concept_image(uploaded_room, room_request or "")
                        st.image(img_bytes, use_column_width=True)
                        st.caption(
                            "The AI kept your room layout and generated updated textiles "
                            "based on your photo and request."
                        )
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT (image-to-image from fixed shelf photo) ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        st.image(
            SHELF_BASE_PATH,
            caption="Market & Place store shelf photo (base image used for AI concept).",
            use_column_width=True,
        )

        shelf_request = st.text_input(
            "Describe how you'd like the shelf styled:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating AI store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(shelf_request or "")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "The AI kept the same store shelf layout and filled it with textiles "
                        "based on your description."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")











