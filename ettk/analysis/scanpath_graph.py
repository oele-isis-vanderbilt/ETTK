def sequence_overlap(seq1: pd.DataFrame, seq2: pd.DataFrame):
    assert len(seq1) == len(seq2)
    overlap = seq1.copy()
    overlap["aoi"][seq1["aoi"] != seq2["aoi"]] = None
    return overlap


def spans_overlap(span1: pd.DataFrame, span2: pd.DataFrame):

    i, j = 0, 0
    overlaps = {"aoi": [], "start_time": [], "end_time": []}
    while True:

        # Find break condition
        if i >= len(span1) - 1 or j >= len(span2):
            break

        # Selecting the data
        span1_row = span1.iloc[i]
        span2_row = span2.iloc[j]

        # First determine if aois match
        if span1_row["aoi"] == span2_row["aoi"]:

            if span1_row["end_time"] <= span2_row["start_time"]:  # Case 1
                logger.debug("Case 1")

            elif (
                span1_row["start_time"] <= span2_row["start_time"]
                and span1_row["end_time"] >= span2_row["start_time"]
                and span1_row["end_time"] <= span2_row["end_time"]
            ):  # Case 2
                overlaps["aoi"].append(span1_row["aoi"])
                overlaps["start_time"].append(span2_row["start_time"])
                overlaps["end_time"].append(span1_row["end_time"])
                logger.debug("Case 2")

            elif (
                span1_row["start_time"] <= span2_row["start_time"]
                and span1_row["end_time"] >= span2_row["end_time"]
            ):  # Case 3
                overlaps["aoi"].append(span1_row["aoi"])
                overlaps["start_time"].append(span2_row["start_time"])
                overlaps["end_time"].append(span2_row["end_time"])
                logger.debug("Case 3")

            elif (
                span2_row["start_time"] <= span1_row["start_time"]
                and span2_row["end_time"] >= span1_row["end_time"]
            ):  # Case 4
                overlaps["aoi"].append(span1_row["aoi"])
                overlaps["start_time"].append(span1_row["start_time"])
                overlaps["end_time"].append(span1_row["end_time"])
                logger.debug("Case 4")

            elif (
                span1_row["start_time"] >= span2_row["start_time"]
                and span1_row["start_time"] <= span2_row["end_time"]
                and span1_row["end_time"] >= span2_row["end_time"]
            ):  # Case 5
                overlaps["aoi"].append(span1_row["aoi"])
                overlaps["start_time"].append(span1_row["start_time"])
                overlaps["end_time"].append(span2_row["end_time"])
                logger.debug("Case 5")

            elif span1_row["start_time"] >= span2_row["end_time"]:  # Case 6
                logger.debug("Case 6")

        # Determine update
        if span1_row["end_time"] > span2_row["end_time"]:
            j += 1
        else:
            i += 1

    return pd.DataFrame(overlaps)


def sequence_to_spans(sequence: pd.DataFrame, by_column: str, time_column: str):

    entries = {by_column: [], "start_time": [], "end_time": []}
    first_sight_row = None
    previous_row = None

    for i, row in sequence.iterrows():

        # Decompose row
        by = row[by_column]
        timestamp = row[time_column]

        if pd.isna(by) or (
            isinstance(previous_row, pd.Series) and by != previous_row[by_column]
        ):
            if (
                isinstance(previous_row, pd.Series)
                and isinstance(first_sight_row, pd.Series)
                and not pd.isna(previous_row[by_column])
            ):

                # Save entry
                entries[by_column].append(previous_row[by_column])
                entries["start_time"].append(first_sight_row[time_column])
                entries["end_time"].append(timestamp)

                # Reset
                first_sight_row = None
        else:
            if not isinstance(first_sight_row, pd.Series):
                first_sight_row = row

        previous_row = row

        # Check if end
        if i == len(sequence) - 1:
            if (
                isinstance(previous_row, pd.Series)
                and isinstance(first_sight_row, pd.Series)
                and not pd.isna(previous_row[by_column])
            ):
                entries[by_column].append(previous_row[by_column])
                entries["start_time"].append(first_sight_row[time_column])
                entries["end_time"].append(timestamp)

    spans = pd.DataFrame(entries)
    return spans


def filter_spans(spans: pd.DataFrame):

    new_spans = {"aoi": [], "start_time": [], "end_time": []}

    i, j = 0, 0
    while True:

        if i >= len(spans):
            break

        # Check if the current and next aoi entries are the same
        if i + 1 < len(spans) and spans.iloc[i]["aoi"] == spans.iloc[i + 1]["aoi"]:

            j = i
            while j < len(spans) - 1:

                # First, check if the aoi matches
                if spans.iloc[j + 1]["aoi"] == spans.iloc[j]["aoi"]:

                    # If they are the same and are close enough, merge them
                    time_delta = (
                        spans.iloc[j + 1]["start_time"] - spans.iloc[j]["end_time"]
                    )
                    if time_delta < MAX_MERGE_TIME_DIFF:
                        j += 1
                    else:
                        break

                else:
                    break

            if j > i:
                new_spans["aoi"].append(spans.iloc[i]["aoi"])
                new_spans["start_time"].append(spans.iloc[i]["start_time"])
                new_spans["end_time"].append(spans.iloc[j]["end_time"])
                i = j + 1
                continue

        # Check if the entry is to small
        duration = spans.iloc[i]["end_time"] - spans.iloc[i]["start_time"]
        if duration > MIN_SPAN_LENGTH:
            new_spans["aoi"].append(spans.iloc[i]["aoi"])
            new_spans["start_time"].append(spans.iloc[i]["start_time"])
            new_spans["end_time"].append(spans.iloc[i]["end_time"])

        # Update the pointer
        i += 1

    # Convert new_spans to DatFrame
    new_spans = pd.DataFrame(new_spans)

    return new_spans


def process_session_data(p_id, tracker):

    # Get data
    cap = cv2.VideoCapture(str(SESSION_PATH / p_id / "scenevideo.mp4"), 0)
    gaze = pd.read_csv(str(SESSION_PATH / p_id / "gaze.csv"))

    writer = cv2.VideoWriter(
        str(
            SESSION_PATH
            / "Analytics"
            / "aoi_rendered_videos"
            / f"rendered_scenevideo_{p_id}.avi"
        ),
        cv2.VideoWriter_fourcc(*"DIVX"),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4))),
    )

    # Process the video
    process_video(p_id, cap, gaze, tracker, writer)

    cap.release()
    writer.release()


def process_aois(meta):

    min_seq_length = float("infinity")
    aoi_sequences = {}
    for p_id in meta["PARTICIPANTS"]:

        # Process the AOI sequences
        aoi_sequence = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "raw_aoi" / f"aoi_{p_id}.csv")
        )
        aoi_sequence["aoi"][aoi_sequence["aoi"] == "laptop"] = "monitor"
        aoi_sequences[p_id] = aoi_sequence.copy()

        # Tracking of the smallest size
        if len(aoi_sequence) < min_seq_length:
            min_seq_length = len(aoi_sequence)

    # Trimmed the sequences to match the same size
    for p_id in aoi_sequences:
        aoi_sequences[p_id] = aoi_sequences[p_id].iloc[0:min_seq_length]

    # Create aoi spans folder if needed
    aoi_spans_folder = SESSION_PATH / "Analytics" / "aoi_spans"
    if not aoi_spans_folder.exists():
        os.mkdir(aoi_spans_folder)

    all_aoi_spans = {}
    for s_name, sequence in aoi_sequences.items():

        save_filename = aoi_spans_folder / f"aoi_spans_{s_name}.csv"

        # Save data
        aoi_spans = sequence_to_spans(
            sequence, by_column="aoi", time_column="timestamp"
        )
        filtered_aoi_spans = filter_spans(aoi_spans)
        filtered_aoi_spans.to_csv(str(save_filename), index=False)

        all_aoi_spans[s_name] = filtered_aoi_spans

    # Then compute the overlap
    for p_id, q_id in [["P1", "P2"], ["P1", "P3"], ["P2", "P3"]]:

        if p_id not in all_aoi_spans or q_id not in all_aoi_spans:
            continue

        concat_name = f"{p_id}-{q_id}"
        save_filename = aoi_spans_folder / f"aoi_spans_{concat_name}.csv"
        overlap = spans_overlap(all_aoi_spans[p_id], all_aoi_spans[q_id])
        overlap = filter_spans(overlap)
        overlap.to_csv(str(save_filename), index=False)
        all_aoi_spans[concat_name] = overlap

    # Then all together
    if "P1" in all_aoi_spans and "P2" in all_aoi_spans and "P3" in all_aoi_spans:
        shared_overlap = spans_overlap(all_aoi_spans["P1-P2"], all_aoi_spans["P1-P3"])
        shared_overlap.to_csv(
            str(aoi_spans_folder / f"aoi_spans_P1-P2-P3.csv"), index=False
        )


def create_timeline(meta):

    # Load the participants data
    # fig, ax = plt.subplots(figsize=(20, 10))
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.grid(False)
    ax.xaxis.set_tick_params(labelsize="large")
    ax.title.set(fontsize=20)
    # fig, ax = plt.subplots(figsize=(10,2))
    p_aoi_bars = []

    if len(meta["PARTICIPANTS"]) == 3:
        # labels = ["P1", "P2", "P3", "P1-P2", "P1-P3", "P2-P3", "P1-P2-P3"]
        labels = ["P1", "P2", "P3"]
        # labels = ["P1"]
    else:
        labels = ["P1", "P2", "P1-P2"]

    for id, p_id in enumerate(labels):

        # Load the data
        if p_id in ["P1", "P2", "P3"]:
            aoi_spans = pd.read_csv(
                str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
            )
        else:
            aoi_spans = pd.read_csv(
                str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
            )

        aoi_spans["duration"] = aoi_spans["end_time"] - aoi_spans["start_time"]
        aoi_spans = colorize("gaze2", aoi_spans, "aoi")

        p_aoi_bars.append(create_modality_bar(ax, aoi_spans, "aoi", 10 * (id + 1)))

    # Adding a shared legend
    categories = [
        "person",
        "bottle",
        "bed",
        "monitor",
        "mouse",
        "thermometer",
        "keyboard",
        "phone/IV pump",
        "paper",
    ]
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(color=GAZE2_COLORSET[category] / 255, label=category)
        )
    ax.legend(
        handles=patches,
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.3),
        bbox_to_anchor=(0.5, -0.3),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=20,
    )

    # Configure the timeline metadata
    plt.xticks(fontsize=20)
    ax.set_xlabel("Time (sec)", fontsize=20)
    ax.set_yticks(
        [10 * (x + 1) + 5 for x in range(len(labels))], labels=labels, fontsize=20
    )
    plt.title(f"AOI Sequences", fontsize=20)
    plt.tight_layout(pad=1)
    plt.savefig(str(SESSION_PATH / "Analytics" / "single_timeline.jpg"))
    plt.show()


def update_graph(span, graph, aoi_spans) -> nx.DiGraph:

    # Determine which objects are observed
    # pdb.set_trace()
    spans = aoi_spans[
        (aoi_spans["end_time"] >= span["start_time"])
        & (aoi_spans["start_time"] < span["end_time"])
    ]
    spans["start_time"] = np.clip(
        spans["start_time"], a_max=None, a_min=span["start_time"]
    )
    spans["end_time"] = np.clip(spans["end_time"], a_max=span["end_time"], a_min=None)
    spans["duration"] = spans["end_time"] - spans["start_time"]

    # Strengthen connections and nodes
    previous_span = None
    added_edges = []
    for i, span in spans.iterrows():

        # Add nodes and their weights
        if span["aoi"] not in graph.nodes():
            graph.add_node(span["aoi"], weight=span["duration"])
        else:
            new_weight = graph.nodes[span["aoi"]]["weight"] + span["duration"]
            graph.nodes[span["aoi"]].update({"weight": new_weight})

        # Add edges and their weights
        if (
            isinstance(previous_span, pd.Series)
            and previous_span["aoi"] != span["aoi"]
            and (previous_span["end_time"] - span["start_time"]) < 15
        ):
            if not graph.has_edge(previous_span["aoi"], span["aoi"]):
                graph.add_edge(previous_span["aoi"], span["aoi"], weight=1)
            else:
                previous_weight = graph.edges[previous_span["aoi"], span["aoi"]][
                    "weight"
                ]
                graph.edges[previous_span["aoi"], span["aoi"]].update(
                    {"weight": previous_weight + 1}
                )

            added_edges.append((previous_span["aoi"], span["aoi"]))

        previous_span = span

    # Decay all nodes and edges
    to_be_removed_nodes = []
    to_be_removed_edges = []

    for node, node_data in graph.nodes(data=True):
        if node not in spans["aoi"]:
            reduced_weight = node_data["weight"] * NODE_DECAY_RATE
            if reduced_weight <= 1:
                # to_be_removed_nodes.append(node)
                graph.nodes[node].update({"weight": 0})
                for x in graph.in_edges(node):
                    if graph.has_edge(*x):
                        to_be_removed_edges.append(x)
                for x in graph.out_edges(node):
                    if graph.has_edge(*x):
                        to_be_removed_edges.append(x)
            else:
                graph.nodes[node].update({"weight": reduced_weight})

    for remove_node in to_be_removed_nodes:
        graph.remove_node(remove_node)

    for node1, node2, edge_data in graph.edges(data=True):
        if (node1, node2) not in added_edges:
            reduced_weight = edge_data["weight"] * EDGE_DECAY_RATE
            if reduced_weight < 0.5:
                to_be_removed_edges.append((node1, node2))
            else:
                graph.edges[node1, node2].update({"weight": reduced_weight})

    for (node1, node2) in to_be_removed_edges:
        if graph.has_edge(node1, node2):
            graph.remove_edge(node1, node2)

    return graph


def create_graph(meta):

    if len(meta["PARTICIPANTS"]) == 3:
        labels = ["P1", "P2", "P3"]
    else:
        labels = ["P1", "P2"]

    # Create the figure and axes
    size = 18
    fig = plt.figure(figsize=(size, size * 0.5625))
    gs = fig.add_gridspec(2, len(labels))
    timeline_ax = fig.add_subplot(gs[1, :])
    axs = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate(labels)}

    # Configure the timeline metadata
    timeline_ax.set_xlabel("Time (sec)", fontsize=15)
    timeline_ax.set_yticks(
        [10 * (x + 1) + 5 for x in range(len(labels))], labels=labels, fontsize=20
    )
    timeline_ax.title.set(fontsize=20)
    timeline_ax.xaxis.set_tick_params(labelsize="large")
    timeline_ax.title.set_text(f"AOI Sequence Timeline")
    timeline_ax.grid(False)
    plt.xticks(fontsize=15)

    # Putting legend for all subplots
    categories = [
        "person",
        "bottle",
        "bed",
        "monitor",
        "mouse",
        "thermometer",
        "keyboard",
        "phone/IV pump",
        "paper",
    ]
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(color=GAZE2_COLORSET[category] / 255, label=category)
        )
    timeline_ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.125),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=15,
    )

    # Creating the static timeline
    p_aoi_bars = []
    all_aoi_spans = {}

    # Creating graphs that will be used later
    graphs = {}
    for k in labels:
        graph = nx.DiGraph()
        graph.add_nodes_from(categories, weight=0)
        graphs[k] = graph

    for id, p_id in enumerate(labels):

        # Load the data
        aoi_spans = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
        )
        aoi_spans["duration"] = aoi_spans["end_time"] - aoi_spans["start_time"]
        aoi_spans = colorize("gaze2", aoi_spans, "aoi")
        all_aoi_spans[p_id] = aoi_spans

        p_aoi_bars.append(
            create_modality_bar(timeline_ax, aoi_spans, "aoi", 10 * (id + 1))
        )

    # Creating directory for temporal_graph_snapshots
    graph_snapshot_dir = SESSION_PATH / "Analytics" / "graph_snapshots"
    if not graph_snapshot_dir.exists():
        os.mkdir(graph_snapshot_dir)

    # Creating the initial rectangle to be drawn in
    # bar_rect = mpatches.Rectangle((0,10), meta["DURATION"] / 200, 9 * len(labels) + len(labels) - 1, color="red")
    bar_rect = mpatches.Rectangle((0, 0), 0, 0, color="red")
    timeline_ax.add_patch(bar_rect)

    def animate(i):

        global graphs

        # Compute time
        previous_time = max(
            0, ((i - ANIMATION_STEP_SIZE) / ANIMATION_RESOLUTION) * meta["DURATION"]
        )
        current_time = min(
            meta["DURATION"], (i / ANIMATION_RESOLUTION) * meta["DURATION"]
        )
        span = {"start_time": previous_time, "end_time": current_time}

        logger.debug(
            f"i: {i}, previous_time: {previous_time}, current_time: {current_time}"
        )

        bar_rect.set_width(meta["DURATION"] / 200)
        bar_rect.set_height(9 * len(labels) + len(labels) - 1)
        bar_rect.set_xy([current_time, 10])

        # Clearning figure
        for ax in axs.values():
            ax.clear()

        for ax_label, ax in axs.items():
            ax.title.set(fontsize=20)
            ax.title.set_text(f"{ax_label} Temporal Graph")
            ax.axis("off")

        # Check if animation just started and ended
        if i == 0:
            graphs = {}
            for k in labels:
                graph = nx.DiGraph()
                graph.add_nodes_from(categories, weight=0)
                graphs[k] = graph
            return [bar_rect]
        elif i >= (ANIMATION_RESOLUTION + ANIMATION_STEP_SIZE):
            graphs = {}
            for k in labels:
                graph = nx.DiGraph()
                graph.add_nodes_from(categories, weight=0)
                graphs[k] = graph
            return [bar_rect]

        # Update the graphs
        for label, p_graph in graphs.items():
            graphs[label] = update_graph(span, p_graph, all_aoi_spans[label])

            # Save the snapshots of the graphs
            with open(graph_snapshot_dir / f"graph_{label}_{i}.pickle", "wb") as f:
                pickle.dump(graphs[label], f)

        # Draw graphs
        drawn_graph_nodes = []
        drawn_graph_labels = []
        drawn_graph_edges = []
        for label, p_graph in graphs.items():

            # Skip empty graphs to avoid NoneType error
            if len(p_graph.nodes()) == 0:
                continue

            # Compute layout
            # pos = nx.spring_layout(p_graph)
            pos = nx.circular_layout(p_graph)

            # Creating graph attributes
            node_weights = [v[1]["weight"] * 10 for v in p_graph.nodes(data=True)]
            node_colors = [GAZE2_COLORSET[v] / 255 for v in p_graph.nodes()]
            edge_weights = [e[2]["weight"] / 2 for e in p_graph.edges(data=True)]

            # Draw functions
            # drawn_graph_edges.extend(
            #     nx.draw_networkx_edges(p_graph, pos, width=edge_weights, ax=axs[label])
            # )
            for edge in p_graph.edges(data=True):
                w = edge[2]["weight"]
                drawn_graph_edges.extend(
                    nx.draw_networkx_edges(
                        p_graph,
                        pos,
                        edgelist=[(edge[0], edge[1])],
                        arrowsize=w,
                        ax=axs[label],
                        width=1,
                    )
                )

            drawn_graph_nodes.append(
                nx.draw_networkx_nodes(
                    p_graph,
                    pos,
                    node_color=node_colors,
                    node_size=node_weights,
                    ax=axs[label],
                )
            )
            # graph_labels = nx.draw_networkx_labels(p_graph, pos, ax=axs[label])
            # drawn_graph_labels.extend([x for x in graph_labels.values()])
            # pdb.set_trace()

        return [bar_rect] + drawn_graph_nodes + drawn_graph_labels + drawn_graph_edges

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=range(
            0, ANIMATION_RESOLUTION + 2 * ANIMATION_STEP_SIZE, ANIMATION_STEP_SIZE
        ),
        # interval=500,
        blit=True,
    )
    plt.show()

    # writervideo = animation.FFMpegWriter(fps=30)
    # ani.save(str(SESSION_PATH / "Analytics" / "temporal_graph.mp4"), writer=writervideo)
    # plt.close()


def compute_graph_attributes(graph) -> Dict:

    metrics = {}

    # Compute metrics for the networks
    # Density: https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html
    metrics["density"] = nx.density(graph)

    # Centrality: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html#:~:text=The%20closeness%20centrality%20is%20normalized,scaled%20by%20that%20parts%20size.
    # centrality = nx.closeness_centrality(graph)
    # max_centrality = max([v for v in centrality.values()])
    # metrics["centrality"] = max_centrality

    centrality = nx.betweenness_centrality(graph, normalized=True, weight="weight")
    metrics["centrality"] = max([v for v in centrality.values()])

    # Efficiency: https://networkx.org/documentation/stable/reference/algorithms/efficiency_measures.html
    metrics["local_efficiency"] = nx.local_efficiency(graph.to_undirected())

    # Small-worldness: https://networkx.org/documentation/stable/reference/algorithms/smallworld.html
    # metrics['small-world'] = nx.sigma(graph.to_undirected())

    # Transitivity: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html
    metrics["transitivity"] = nx.transitivity(graph)

    # Global efficiency: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.efficiency_measures.global_efficiency.html#networkx.algorithms.efficiency_measures.global_efficiency
    metrics["global efficiency"] = nx.global_efficiency(graph.to_undirected())

    return metrics


def save_model(model, model_name: str, dir: pathlib.Path):
    # np.save(str(dir / f"{model_name}_weights"), model.weights_, allow_pickle=False)
    # np.save(str(dir / f"{model_name}_means"), model.means_, allow_pickle=False)
    # np.save(
    #     str(dir / f"{model_name}_covariances"), model.covariances_, allow_pickle=False
    # )
    with open(dir / f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(model_name: str, dir: pathlib.Path):

    with open(dir / f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    # means = np.load(str(dir / f"{model_name}_means.npy"))
    # covar = np.load(str(dir / f"{model_name}_covariances.npy"))

    # model = sklearn.mixture.GaussianMixture(
    #     n_components=len(means), covariance_type="full"
    # )

    # model.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    # model.weights_ = np.load(str(dir / f"{model_name}_weights.npy"))
    # model.means_ = means
    # model.covariances_ = covar

    return model


def graph_embedding(graph: nx.Graph):

    edge_weights_vector = np.asarray(
        nx.to_numpy_matrix(graph, weight="weight")
    ).flatten()
    norm_edge_weights_vector = edge_weights_vector / max(1, np.sum(edge_weights_vector))
    norm_edge_weights_vector = np.squeeze(norm_edge_weights_vector)

    node_weights = np.array([v[1]["weight"] for v in graph.nodes(data=True)])
    norm_node_weights = node_weights / max(1, np.sum(node_weights))

    embedding = np.concatenate([norm_edge_weights_vector, norm_node_weights], axis=0)

    return embedding


def graph_feature_creation(meta):

    # Load the graph data
    graph_snapshots = []
    graph_snapshot_dir = SESSION_PATH / "Analytics" / "graph_snapshots"
    for graph_snapshot_file in graph_snapshot_dir.iterdir():
        with open(graph_snapshot_file, "rb") as f:
            graph = pickle.load(f)
        graph_snapshots.append(graph)

    # Compute graph metrics
    graphs_embeds = []
    for graph in graph_snapshots:
        graph_embed = graph_embedding(graph)
        graphs_embeds.append(graph_embed)

    graphs_embeds_dir = SHARED_DIR / "graph_embeddings"
    if not graphs_embeds_dir.exists():
        os.mkdir(graphs_embeds_dir)

    # Then concat the entire matrix
    graphs_embeds_stack = np.stack(graphs_embeds)
    with open(graphs_embeds_dir / f"{SESSION_PATH.stem}_graph_embeds.npy", "wb") as f:
        np.save(f, graphs_embeds_stack)


def embed2graph(embed) -> nx.DiGraph:

    categories = [
        "person",
        "bottle",
        "bed",
        "monitor",
        "mouse",
        "thermometer",
        "keyboard",
        "phone/IV pump",
        "paper",
    ]
    c_size = len(categories)

    # Get the edges information
    norm_edge_weights_vector = embed[: c_size * c_size]
    norm_edge_weights_vector = norm_edge_weights_vector / sum(norm_edge_weights_vector)
    adjacency_matrix = norm_edge_weights_vector.reshape((c_size, c_size))

    # Apply a cut off for edges
    adjacency_matrix = np.where(adjacency_matrix > EDGE_CUTTOFF, adjacency_matrix, 0)

    graph = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph, {i: c for i, c in enumerate(categories)})

    # Get the node information
    norm_node_weights_vector = embed[c_size * c_size :]
    norm_node_weights_vector = norm_node_weights_vector / sum(norm_node_weights_vector)
    norm_node_weights_vector = np.where(
        norm_node_weights_vector > NODE_CUTTOFF, norm_node_weights_vector, 0
    )

    to_be_removed_edges = []
    for node_name, node_weight in zip(categories, norm_node_weights_vector):

        # Adding the weight
        graph.nodes[node_name]["weight"] = node_weight

        if node_weight == 0:
            for x in graph.in_edges(node_name):
                if graph.has_edge(*x):
                    to_be_removed_edges.append(x)
            for x in graph.out_edges(node_name):
                if graph.has_edge(*x):
                    to_be_removed_edges.append(x)

    for (node1, node2) in to_be_removed_edges:
        if graph.has_edge(node1, node2):
            graph.remove_edge(node1, node2)

    return graph


def optimal_clustering_search(meta):

    # Load all the graph embeddings
    graphs_embeds_dir = SHARED_DIR / "graph_embeddings"
    graphs_embeds = []
    for embeddings_file in graphs_embeds_dir.iterdir():
        with open(embeddings_file, "rb") as f:
            embedding = np.load(f)
            graphs_embeds.append(embedding)

    # Create a complete input feature set
    graph_data = np.concatenate(graphs_embeds)

    # Create the model
    # model = sklearn.mixture.GaussianMixture()
    model = sklearn.cluster.KMeans(N_CLUSTERS)

    # Find the best number of clusters
    # visualizer = KElbowVisualizer(model, k=(2,10), timings=True)
    # visualizer.fit(graph_data)
    # visualizer.show()
    model.fit(graph_data)

    # Save the model
    model_save_path = SHARED_DIR / "cluster_model"
    if not model_save_path.exists():
        os.mkdir(model_save_path)
    save_model(model, "kmeans", model_save_path)


def graph_clustering(meta):

    # Get the model
    model_save_path = SHARED_DIR / "cluster_model"
    model = load_model("kmeans", model_save_path)

    # Load the graph data
    graph_snapshots_dict = collections.defaultdict(dict)
    graph_snapshot_dir = SESSION_PATH / "Analytics" / "graph_snapshots"
    for graph_snapshot_file in graph_snapshot_dir.iterdir():
        with open(graph_snapshot_file, "rb") as f:
            graph = pickle.load(f)

        # Determine which participant and time
        # f"graph_{label}_{i}.pickle"
        p_id = re.search(r"graph_(.*?)_", graph_snapshot_file.name).group(1)
        i = re.search(p_id + r"_(.*?)\.pickle", graph_snapshot_file.name).group(1)

        graph_snapshots_dict[p_id][i] = graph

    # Then construct their data!
    graph_info_dict = SESSION_PATH / "Analytics" / "graph_features"
    if not graph_info_dict.exists():
        os.mkdir(graph_info_dict)

    for p_id, snapshots_dict in graph_snapshots_dict.items():

        all_p_graphs = collections.defaultdict(list)
        for i, graph in sorted(snapshots_dict.items()):

            # Get graph data
            # graph_metrics = compute_graph_attributes(graph)
            graph_embed = graph_embedding(graph)
            graph_id = model.predict(graph_embed.reshape((1, -1)))[0]

            # Store it!
            all_p_graphs["time_id"].append(int(i))
            all_p_graphs["timestamp"].append(
                float(meta["DURATION"]) * (int(i) / ANIMATION_RESOLUTION)
            )
            all_p_graphs["cluster_id"].append(graph_id)

            # for metric, metric_value in graph_metrics.items():
            #     all_p_graphs[metric].append(metric_value)

        # Save the data
        all_p_graphs_filename = graph_info_dict / f"{p_id}_features.csv"
        all_p_graphs_df = pd.DataFrame(all_p_graphs)
        all_p_graphs_df.time_id = all_p_graphs_df.time_id.astype(int)
        all_p_graphs_df = all_p_graphs_df.sort_values(by=["time_id"])
        all_p_graphs_df.to_csv(all_p_graphs_filename, index=False)


def cluster_average(meta):

    model_save_path = SHARED_DIR / "cluster_model"
    model = load_model("kmeans", model_save_path)

    # Grab the means and construct graphs based on it
    embeds = np.split(model.cluster_centers_, model.cluster_centers_.shape[0])

    # Create the general figure
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3)
    axs = [
        fig.add_subplot(gs[y, x]) for y, x in ([0, 0], [0, 1], [0, 2], [1, 0], [1, 1])
    ]

    # Convert embeddings to graphs
    all_graph_metrics = collections.defaultdict(list)
    for i, embed in enumerate(embeds):
        graph = embed2graph(np.squeeze(embed))

        # Creating graph attributes
        node_weights = [v[1]["weight"] * 1000 for v in graph.nodes(data=True)]
        node_colors = [GAZE2_COLORSET[v] / 255 for v in graph.nodes()]
        edge_weights = [e[2]["weight"] * 50 for e in graph.edges(data=True)]

        arrowstyle = mpatches.ArrowStyle(
            "Simple", head_length=0.4, head_width=0.4, tail_width=0.4
        )

        # Create visualization of the graphs
        # fig, ax = plt.subplots(figsize=(6, 6))
        axs[i].grid(False)
        # plt.title(f"Cluster {i} Graph Average")
        axs[i].set_title(f"Cluster {i+1} Graph Average")
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(
            graph, pos, node_size=node_weights, node_color=node_colors, ax=axs[i]
        )
        for edge in graph.edges(data=True):
            w = edge[2]["weight"] * 100
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(edge[0], edge[1])], arrowsize=w, ax=axs[i]
            )

        # Save the graphs
        # plt.savefig(str(SHARED_DIR / "cluster_model" / f"cluster_{i}.png"))
        # plt.clf()

        # Compute graph metrics
        graph_metrics = compute_graph_attributes(graph)
        for k, v in graph_metrics.items():
            all_graph_metrics[k].append(v)

    # Add the legend
    categories = [
        "person",
        "bottle",
        "bed",
        "monitor",
        "mouse",
        "thermometer",
        "keyboard",
        "phone/IV pump",
        "paper",
    ]
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(color=GAZE2_COLORSET[category] / 255, label=category)
        )
    axs[2].legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.125),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    # plt.tight_layout(pad=0)
    # plt.show()
    plt.savefig(str(SHARED_DIR / "cluster_model" / "graph_cluster_averages.png"))

    all_graph_metrics_df = pd.DataFrame(all_graph_metrics)
    all_graph_metrics_df.to_csv(
        str(SHARED_DIR / "cluster_model" / "cluster_graph_metrics.csv"), index=False
    )


def graph_clustering_visualization(meta):

    # Load the participants data
    fig, ax = plt.subplots(figsize=(20, 10))
    p_bars = []

    if len(meta["PARTICIPANTS"]) == 3:
        participants = ["P1", "P2", "P3"]
        labels = ["P1", "P1-C", "P2", "P2-C", "P3", "P3-C"]
    else:
        participants = ["P1", "P2"]
        labels = ["P1", "P1-C", "P2", "P2-C"]

    for id, p_id in enumerate(participants):

        # Load the data
        graph_features = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "graph_features" / f"{p_id}_features.csv")
        )

        # Convert cluster to span
        cluster_span = sequence_to_spans(
            graph_features, by_column="cluster_id", time_column="timestamp"
        )

        cluster_span["duration"] = cluster_span["end_time"] - cluster_span["start_time"]
        cluster_span = colorize("clusters", cluster_span, "cluster_id")

        aoi_spans = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
        )
        aoi_spans["duration"] = aoi_spans["end_time"] - aoi_spans["start_time"]
        aoi_spans = colorize("gaze2", aoi_spans, "aoi")

        p_bars.append(create_modality_bar(ax, aoi_spans, "aoi", 20 * (id)))
        p_bars.append(
            create_modality_bar(ax, cluster_span, "cluster_id", 20 * (id) + 10)
        )

    # Adding a shared legend
    categories = list(range(N_CLUSTERS))
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(color=CLUSTER_COLORSET[category] / 255, label=category)
        )
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    # Configure the timeline metadata
    ax.set_xlabel("Time (sec)")
    ax.set_yticks([10 * (x) + 5 for x in range(len(labels))], labels=labels)
    plt.title(f"Cluster Sequences")
    plt.savefig(str(SESSION_PATH / "Analytics" / "graph_clusters.jpg"))
    plt.clf()
    # plt.show()


def event_timeline_with_cluster(meta):

    # Load the participants data
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.xaxis.set_tick_params(labelsize="large")
    p_bars = []

    if len(meta["PARTICIPANTS"]) == 3:
        participants = ["P1", "P2", "P3"]
        labels = ["P1", "P1-C", "P2", "P2-C", "P3", "P3-C"]
    else:
        participants = ["P1", "P2"]
        labels = ["P1", "P1-C", "P2", "P2-C"]

    for id, p_id in enumerate(participants):

        # Load the data
        graph_features = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "graph_features" / f"{p_id}_features.csv")
        )

        # Convert cluster to span
        cluster_span = sequence_to_spans(
            graph_features, by_column="cluster_id", time_column="timestamp"
        )

        cluster_span["duration"] = cluster_span["end_time"] - cluster_span["start_time"]
        cluster_span = colorize("clusters", cluster_span, "cluster_id")

        aoi_spans = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
        )
        aoi_spans["duration"] = aoi_spans["end_time"] - aoi_spans["start_time"]
        aoi_spans = colorize("gaze2", aoi_spans, "aoi")

        p_bars.append(create_modality_bar(ax, aoi_spans, "aoi", 20 * (id) + 5))
        p_bars.append(
            create_modality_bar(ax, cluster_span, "cluster_id", 20 * (id) + 15)
        )

    # Add event timeline
    event_times = [0, 40, 168, 212, 319, 371, 426, 771, 866, 999, 1404]
    event_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]

    # levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(event_times)/6)))[:len(event_times)]
    levels = [64 for x in range(len(event_times))]

    ax.vlines(event_times, 0, levels, color="k")
    ax.plot(
        event_times,
        np.zeros_like(event_times),
        marker="o",
        color="k",
        markerfacecolor="w",
        markeredgewidth=1.5,
        markeredgecolor=(0, 0, 0, 1),
    )

    for d, l, r in zip(event_times, levels, event_names):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if l > 0 else "top",
            fontsize=20,
        )

    # Adding a shared legend
    categories = list(range(N_CLUSTERS))
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(
                color=CLUSTER_COLORSET[category] / 255,
                label=f"{category+1}: {CLUSTER_MAPPING[category]}",
            )
        )
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.064),
        fancybox=True,
        shadow=True,
        ncol=3,
        fontsize=15,
    )

    # Configure the timeline metadata
    ax.grid(False)
    plt.xticks(fontsize=15)
    ax.set_xlabel("Time (sec)", fontsize=15)
    new_labels = ["Events"] + labels
    ax.set_yticks(
        [0] + [10 * (x) + 10 for x in range(len(new_labels) - 1)],
        labels=new_labels,
        fontsize=20,
    )
    plt.title(f"Cluster Sequences", fontsize=20)
    plt.savefig(str(SESSION_PATH / "Analytics" / "graph_clusters_with_events.jpg"))
    plt.show()
    # plt.clf()


def graph_similarity(meta):

    # Load the participants data
    fig, ax = plt.subplots(figsize=(20, 10))
    p_bars = []

    if len(meta["PARTICIPANTS"]) == 3:
        labels = ["SIM", "P1", "P2", "P3"]
        participants = ["P1", "P2", "P3"]
        possible_comparison = [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]
    else:
        labels = ["SIM", "P1", "P2"]
        participants = ["P1", "P2"]
        possible_comparison = [("P1", "P2")]

    # Get all the graphs features
    all_graph_features = {}
    for id, p_id in enumerate(participants):
        graph_features = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "graph_features" / f"{p_id}_features.csv")
        )
        all_graph_features[p_id] = graph_features

    # Then couple each possible pair and check for similiarity through MSE
    compare_diffs = {}
    for comparison in possible_comparison:
        a, b = comparison
        graph_a, graph_b = all_graph_features[a], all_graph_features[b]

        diffs = []
        for column_name in graph_a.columns:
            if column_name in ["timestamp", "time_id", "cluster_id"]:
                continue

            diff = 1 - np.abs((graph_a[column_name] - graph_b[column_name]))
            diffs.append(diff)

        total_diffs = sum([np.power(x, 2) for x in diffs])
        compare_diffs[comparison] = total_diffs

    # Draw the xy lines
    for pair, data in compare_diffs.items():
        color = PAIR_COLORSET[f"({pair[0]},{pair[1]})"] / 255
        ax.plot(
            all_graph_features["P1"]["timestamp"].to_numpy(),
            data.to_numpy(),
            color=color,
        )

    for id, p_id in enumerate(participants):

        # Load the data
        aoi_spans = pd.read_csv(
            str(SESSION_PATH / "Analytics" / "aoi_spans" / f"aoi_spans_{p_id}.csv")
        )
        aoi_spans["duration"] = aoi_spans["end_time"] - aoi_spans["start_time"]
        aoi_spans = colorize("gaze2", aoi_spans, "aoi")

        p_bars.append(create_modality_bar(ax, aoi_spans, "aoi", 10 * (id + 1)))

    # Adding a shared legend
    categories = list(compare_diffs.keys())
    patches = []
    for category in categories:
        patches.append(
            mpatches.Patch(
                color=PAIR_COLORSET[f"({category[0]},{category[1]})"] / 255,
                label=str(category),
            )
        )
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    # Configure the timeline metadata
    ax.set_xlabel("Time (sec)")
    ax.set_yticks([10 * (x) + 5 for x in range(len(labels))], labels=labels)
    plt.title(f"Graph Similiarty")
    plt.savefig(str(SESSION_PATH / "Analytics" / "graph_similarity.jpg"))
    plt.show()
