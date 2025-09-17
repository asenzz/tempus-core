--
-- PostgreSQL database dump
--

-- Dumped from database version 16.1
-- Dumped by pg_dump version 16.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: cleanup_queue(regclass, timestamp without time zone[]); Type: FUNCTION; Schema: public; Owner: svrwave
--

CREATE FUNCTION public.cleanup_queue(tbl regclass, value_times timestamp without time zone[]) RETURNS void
    LANGUAGE plpgsql
    AS $$
        BEGIN
                EXECUTE format('delete from %s q1 where exists (select 1 from unnest(%L::timestamp[]) tm where tm = q1.value_time)', tbl, value_times);
            END
            $$;


ALTER FUNCTION public.cleanup_queue(tbl regclass, value_times timestamp without time zone[]) OWNER TO svrwave;

--
-- Name: mark_interval_reconciled(text, timestamp without time zone, timestamp without time zone); Type: FUNCTION; Schema: public; Owner: svrwave
--

CREATE FUNCTION public.mark_interval_reconciled("tableName" text, "startTime" timestamp without time zone, "endTime" timestamp without time zone) RETURNS void
    LANGUAGE plpgsql
    AS $$DECLARE
    rownum integer;
    cur_start  timestamp without time zone;
    cur_end  timestamp without time zone;
BEGIN
    SELECT INTO rownum count(1)
    FROM input_queue_reconciled_hours
    WHERE table_name = "tableName";

    IF rownum > 0 THEN
	SELECT into cur_start min(start_time)
	FROM input_queue_reconciled_hours
        WHERE table_name = "tableName";
	IF "startTime" < cur_start THEN
	    cur_start := "startTime";
	END IF;

	SELECT into cur_end max(end_time)
	FROM input_queue_reconciled_hours
        WHERE table_name = "tableName";
	IF "endTime" > cur_end THEN
	    cur_end := "endTime";
	END IF;
        
	update input_queue_reconciled_hours 
	set last_request_time = now(), start_time = cur_start, end_time=cur_end
	WHERE table_name = "tableName";
    ELSE
	insert into input_queue_reconciled_hours (table_name, start_time, end_time, last_request_time)
	values ("tableName", "startTime", "endTime", now());
    END IF;
END;$$;


ALTER FUNCTION public.mark_interval_reconciled("tableName" text, "startTime" timestamp without time zone, "endTime" timestamp without time zone) OWNER TO svrwave;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: autotune_tasks; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.autotune_tasks (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    result_dataset_id bigint,
    creation_time timestamp without time zone DEFAULT now() NOT NULL,
    done_time timestamp without time zone,
    parameters json NOT NULL,
    start_train_time timestamp without time zone NOT NULL,
    end_train_time timestamp without time zone NOT NULL,
    start_tuning_time timestamp without time zone NOT NULL,
    end_tuning_time timestamp without time zone NOT NULL,
    vp_sliding_direction smallint,
    vp_slide_count smallint,
    vp_slide_period_sec bigint,
    pso_best_points_counter smallint,
    pso_iteration_number smallint,
    pso_particles_number smallint,
    pso_topology smallint,
    nm_max_iteration_number smallint,
    nm_tolerance double precision,
    status smallint DEFAULT 0,
    mse double precision DEFAULT '-1.0'::numeric
);


ALTER TABLE public.autotune_tasks OWNER TO svrwave;

--
-- Name: autotune_tasks_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.autotune_tasks_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.autotune_tasks_seq OWNER TO svrwave;

--
-- Name: datasets; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.datasets (
    id bigint NOT NULL,
    dataset_name text NOT NULL,
    user_name text NOT NULL,
    main_input_queue_table_name text NOT NULL,
    aux_input_queues_table_names text[],
    priority smallint NOT NULL,
    description text,
    levels smallint NOT NULL,
    deconstruction text NOT NULL,
    max_gap interval NOT NULL,
    is_active boolean DEFAULT false,
    gradients smallint DEFAULT 1 NOT NULL,
    max_chunk_size integer DEFAULT 4000 NOT NULL,
    multistep smallint DEFAULT 1 NOT NULL
);


ALTER TABLE public.datasets OWNER TO svrwave;

--
-- Name: datasets_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.datasets_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.datasets_seq OWNER TO svrwave;

--
-- Name: decon_queue_template; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.decon_queue_template (
    value_time timestamp without time zone NOT NULL,
    update_time timestamp without time zone DEFAULT now() NOT NULL,
    tick_volume double precision DEFAULT 1.0 NOT NULL
);


ALTER TABLE public.decon_queue_template OWNER TO svrwave;

--
-- Name: decon_queues; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.decon_queues (
    table_name text NOT NULL,
    input_queue_table_name text NOT NULL,
    input_queue_column_name text NOT NULL,
    dataset_id bigint NOT NULL
);


ALTER TABLE public.decon_queues OWNER TO svrwave;

--
-- Name: dq_scaling_factors; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.dq_scaling_factors (
    id bigint NOT NULL,
    model_id bigint NOT NULL,
    input_queue_table_name text NOT NULL,
    input_queue_column_name text NOT NULL,
    level smallint NOT NULL,
    step smallint DEFAULT 0 NOT NULL,
    scaling_factor_features double precision DEFAULT 1 NOT NULL,
    scaling_factor_labels double precision DEFAULT 1 NOT NULL,
    dc_offset_features double precision DEFAULT 0 NOT NULL,
    dc_offset_labels double precision DEFAULT 0 NOT NULL,
    gradient smallint DEFAULT 0 NOT NULL,
    chunk smallint DEFAULT 0 NOT NULL
);


ALTER TABLE public.dq_scaling_factors OWNER TO svrwave;

--
-- Name: dq_scaling_factors_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.dq_scaling_factors_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.dq_scaling_factors_seq OWNER TO svrwave;

--
-- Name: ensembles; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.ensembles (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    decon_queue_table_name text,
    aux_decon_queues_table_names text[]
);


ALTER TABLE public.ensembles OWNER TO svrwave;

--
-- Name: ensembles_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.ensembles_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.ensembles_seq OWNER TO svrwave;

--
-- Name: input_queue_reconciled_hours; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.input_queue_reconciled_hours (
    table_name text NOT NULL,
    start_time timestamp without time zone,
    end_time timestamp without time zone,
    last_request_time timestamp without time zone
);


ALTER TABLE public.input_queue_reconciled_hours OWNER TO svrwave;

--
-- Name: TABLE input_queue_reconciled_hours; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON TABLE public.input_queue_reconciled_hours IS 'Contains the time ranges that could not be uploaded to the input queue. Such time ranges may be the exchange closed hours as well as technical errors';


--
-- Name: COLUMN input_queue_reconciled_hours.last_request_time; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.input_queue_reconciled_hours.last_request_time IS 'Last time this interval was requested and failed.';


--
-- Name: input_queue_template; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.input_queue_template (
    value_time timestamp without time zone NOT NULL,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    tick_volume double precision DEFAULT 1 NOT NULL
);


ALTER TABLE public.input_queue_template OWNER TO svrwave;

--
-- Name: input_queues; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.input_queues (
    table_name text NOT NULL,
    logical_name text NOT NULL,
    user_name text NOT NULL,
    description text,
    resolution interval NOT NULL,
    legal_time_deviation interval NOT NULL,
    timezone text DEFAULT 'UTC'::text,
    value_columns text[] NOT NULL,
    missing_hours_retention interval DEFAULT '14 days'::interval NOT NULL,
    uses_fix_connection boolean DEFAULT false NOT NULL
);


ALTER TABLE public.input_queues OWNER TO svrwave;

--
-- Name: iq_scaling_factors; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.iq_scaling_factors (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    input_queue_table_name text NOT NULL,
    scaling_factor double precision DEFAULT 1.0,
    dc_offset double precision DEFAULT 0 NOT NULL,
    input_queue_column_name text NOT NULL
);


ALTER TABLE public.iq_scaling_factors OWNER TO svrwave;

--
-- Name: iq_scaling_factors_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.iq_scaling_factors_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.iq_scaling_factors_seq OWNER TO svrwave;

--
-- Name: models; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.models (
    id bigint NOT NULL,
    ensemble_id bigint NOT NULL,
    decon_level smallint NOT NULL,
    learning_levels smallint[],
    model_binary bytea,
    last_modified_time timestamp without time zone,
    last_modeled_value_time timestamp without time zone,
    norm_mean_coef double precision,
    norm_range_coef double precision,
    gradients smallint DEFAULT 1 NOT NULL
);


ALTER TABLE public.models OWNER TO svrwave;

--
-- Name: models_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.models_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.models_seq OWNER TO svrwave;

--
-- Name: multival_requests; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.multival_requests (
    request_id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    user_name text,
    request_time timestamp without time zone DEFAULT now() NOT NULL,
    value_time_start timestamp without time zone NOT NULL,
    value_time_end timestamp without time zone NOT NULL,
    resolution integer DEFAULT 60,
    value_columns text[],
    processed boolean DEFAULT false
);


ALTER TABLE public.multival_requests OWNER TO svrwave;

--
-- Name: TABLE multival_requests; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON TABLE public.multival_requests IS 'Contains multiple value requests.';


--
-- Name: COLUMN multival_requests.value_time_start; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.multival_requests.value_time_start IS 'The first requested time.';


--
-- Name: COLUMN multival_requests.resolution; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.multival_requests.resolution IS 'One interval length in seconds.';


--
-- Name: multival_results; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.multival_results (
    response_id bigint NOT NULL,
    request_id bigint NOT NULL,
    value_time timestamp without time zone NOT NULL,
    value_column text NOT NULL,
    value double precision NOT NULL
);


ALTER TABLE public.multival_results OWNER TO svrwave;

--
-- Name: parmtune_decrement_tasks; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.parmtune_decrement_tasks (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    start_task_time timestamp without time zone NOT NULL,
    end_task_time timestamp without time zone,
    start_train_time timestamp without time zone NOT NULL,
    end_train_time timestamp without time zone NOT NULL,
    start_validation_time timestamp without time zone NOT NULL,
    end_validation_time timestamp without time zone NOT NULL,
    parameters json NOT NULL,
    status smallint,
    decrement_step text NOT NULL,
    vp_sliding_direction smallint,
    vp_slide_count smallint,
    vp_slide_period_sec bigint,
    "values" json,
    suggested_value json
);


ALTER TABLE public.parmtune_decrement_tasks OWNER TO svrwave;

--
-- Name: COLUMN parmtune_decrement_tasks.id; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks.id IS 'Task id number.';


--
-- Name: COLUMN parmtune_decrement_tasks.end_task_time; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks.end_task_time IS 'When time is set, task has ended.';


--
-- Name: COLUMN parmtune_decrement_tasks.start_train_time; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks.start_train_time IS 'Start of training range.';


--
-- Name: COLUMN parmtune_decrement_tasks.end_train_time; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks.end_train_time IS 'End of training range.';


--
-- Name: COLUMN parmtune_decrement_tasks.status; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks.status IS 'Task status and resultant error code if done.';


--
-- Name: COLUMN parmtune_decrement_tasks."values"; Type: COMMENT; Schema: public; Owner: svrwave
--

COMMENT ON COLUMN public.parmtune_decrement_tasks."values" IS 'Contains a vector of MSE vs observations count, and the recommended observation count for every level of the dataset predicted input queue.';


--
-- Name: parmtune_decrement_tasks_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.parmtune_decrement_tasks_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.parmtune_decrement_tasks_seq OWNER TO svrwave;

--
-- Name: prediction_tasks; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.prediction_tasks (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    start_time timestamp without time zone NOT NULL,
    end_time timestamp without time zone NOT NULL,
    start_prediction_time timestamp without time zone NOT NULL,
    end_prediction_time timestamp without time zone NOT NULL,
    status smallint DEFAULT 0,
    mse double precision DEFAULT '-1.0'::numeric
);


ALTER TABLE public.prediction_tasks OWNER TO svrwave;

--
-- Name: prediction_tasks_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.prediction_tasks_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.prediction_tasks_seq OWNER TO svrwave;

--
-- Name: request_id_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.request_id_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.request_id_seq OWNER TO svrwave;

--
-- Name: result_id_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.result_id_seq
    START WITH 16932
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.result_id_seq OWNER TO svrwave;

--
-- Name: svr_parameters; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.svr_parameters (
    id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    input_queue_table_name text NOT NULL,
    input_queue_column_name text NOT NULL,
    decon_level smallint NOT NULL,
    svr_c double precision NOT NULL,
    svr_epsilon double precision NOT NULL,
    svr_kernel_param double precision NOT NULL,
    svr_kernel_param2 double precision NOT NULL,
    svr_decremental_distance bigint NOT NULL,
    svr_adjacent_levels_ratio double precision NOT NULL,
    svr_kernel_type smallint NOT NULL,
    lag_count smallint NOT NULL,
    levels smallint DEFAULT 1 NOT NULL,
    step smallint DEFAULT 0 NOT NULL,
    chunk_ix smallint DEFAULT 0 NOT NULL,
    grad_level smallint DEFAULT 0 NOT NULL,
    svr_kernel_param3 double precision DEFAULT 0 NOT NULL
);


ALTER TABLE public.svr_parameters OWNER TO svrwave;

--
-- Name: svr_parameters_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.svr_parameters_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.svr_parameters_seq OWNER TO svrwave;

--
-- Name: user_datasets; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.user_datasets (
    user_id bigint NOT NULL,
    dataset_id bigint NOT NULL
);


ALTER TABLE public.user_datasets OWNER TO svrwave;

--
-- Name: userauth; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.userauth (
    user_id bigint NOT NULL,
    username text NOT NULL,
    password text,
    email text,
    name text,
    role text,
    priority smallint DEFAULT 2 NOT NULL
);


ALTER TABLE public.userauth OWNER TO svrwave;

--
-- Name: userauth_seq; Type: SEQUENCE; Schema: public; Owner: svrwave
--

CREATE SEQUENCE public.userauth_seq
    START WITH 100
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.userauth_seq OWNER TO svrwave;

--
-- Name: v_user_datasets; Type: VIEW; Schema: public; Owner: svrwave
--

CREATE VIEW public.v_user_datasets AS
 SELECT DISTINCT id,
    dataset_name,
    user_name,
    main_input_queue_table_name,
    aux_input_queues_table_names,
    priority,
    description,
    swt_levels,
    swt_wavelet_name,
    max_gap,
    is_active,
    linked_user_name,
    user_priority
   FROM ( SELECT ds.id,
            ds.dataset_name,
            ds.user_name,
            ds.main_input_queue_table_name,
            ds.aux_input_queues_table_names,
            ds.priority,
            ds.description,
            ds.levels AS swt_levels,
            ds.deconstruction AS swt_wavelet_name,
            ds.max_gap,
            ds.is_active,
            ua.username AS linked_user_name,
            ua.priority AS user_priority
           FROM ((public.datasets ds
             JOIN public.user_datasets ud ON ((ds.id = ud.dataset_id)))
             JOIN public.userauth ua ON ((ua.user_id = ud.user_id)))
          WHERE (ds.is_active = true)
        UNION ALL
         SELECT ds.id,
            ds.dataset_name,
            ds.user_name,
            ds.main_input_queue_table_name,
            ds.aux_input_queues_table_names,
            ds.priority,
            ds.description,
            ds.levels AS swt_levels,
            ds.deconstruction AS swt_wavelet_name,
            ds.max_gap,
            ds.is_active,
            ua.username AS linked_user_name,
            ua.priority AS user_priority
           FROM (public.datasets ds
             JOIN public.userauth ua ON ((ua.username = ds.user_name)))
          WHERE (ds.is_active = true)) all_ds
  ORDER BY user_priority DESC, priority DESC;


ALTER VIEW public.v_user_datasets OWNER TO svrwave;

--
-- Name: w_scaling_factors; Type: TABLE; Schema: public; Owner: asenzz
--

CREATE TABLE public.w_scaling_factors (
    id bigint NOT NULL,
    dataset_id bigint,
    step smallint,
    scaling_factor double precision DEFAULT 1 NOT NULL,
    dc_offset double precision DEFAULT 0 NOT NULL
);


ALTER TABLE public.w_scaling_factors OWNER TO asenzz;

--
-- Data for Name: autotune_tasks; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.autotune_tasks (id, dataset_id, result_dataset_id, creation_time, done_time, parameters, start_train_time, end_train_time, start_tuning_time, end_tuning_time, vp_sliding_direction, vp_slide_count, vp_slide_period_sec, pso_best_points_counter, pso_iteration_number, pso_particles_number, pso_topology, nm_max_iteration_number, nm_tolerance, status, mse) FROM stdin;
\.


--
-- Data for Name: datasets; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.datasets (id, dataset_name, user_name, main_input_queue_table_name, aux_input_queues_table_names, priority, description, levels, deconstruction, max_gap, is_active, gradients, max_chunk_size, multistep) FROM stdin;
100	xauusd	svrwave	q_svrwave_xauusd_avg_3600	{q_svrwave_xauusd_avg_1}	2		64	cvmd	92:00:00	t	1	4000	1
\.


--
-- Data for Name: decon_queue_template; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.decon_queue_template (value_time, update_time, tick_volume) FROM stdin;
\.


--
-- Data for Name: decon_queues; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.decon_queues (table_name, input_queue_table_name, input_queue_column_name, dataset_id) FROM stdin;
z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid	q_svrwave_xauusd_avg_3600	xauusd_avg_bid	100
z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid	q_svrwave_xauusd_avg_1	xauusd_avg_bid	100
\.


--
-- Data for Name: dq_scaling_factors; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.dq_scaling_factors (id, model_id, input_queue_table_name, input_queue_column_name, level, step, scaling_factor_features, scaling_factor_labels, dc_offset_features, dc_offset_labels, gradient, chunk) FROM stdin;
\.


--
-- Data for Name: ensembles; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.ensembles (id, dataset_id, decon_queue_table_name, aux_decon_queues_table_names) FROM stdin;
100	100	z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid	{z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid}
\.


--
-- Data for Name: input_queue_reconciled_hours; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.input_queue_reconciled_hours (table_name, start_time, end_time, last_request_time) FROM stdin;
\.


--
-- Data for Name: input_queue_template; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.input_queue_template (value_time, update_time, tick_volume) FROM stdin;
\.


--
-- Data for Name: input_queues; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.input_queues (table_name, logical_name, user_name, description, resolution, legal_time_deviation, timezone, value_columns, missing_hours_retention, uses_fix_connection) FROM stdin;
q_svrwave_xauusd_avg_3600	xauusd_avg	svrwave	q_svrwave_xauusd_avg_3600	01:00:00	00:00:05	EEST	{xauusd_avg_bid}	14 days	f
q_svrwave_xauusd_avg_1	xauusd_avg	svrwave	q_svrwave_xauusd_avg_1	00:00:01	00:00:00	EEST	{xauusd_avg_bid}	14 days	f
q_svrwave_test_xauusd_avg_3600	xauusd_avg	svrwave	q_svrwave_xauusd_avg_3600	01:00:00	00:00:05	EEST	{xauusd_avg_bid}	14 days	f
q_svrwave_test_xauusd_avg_1	xauusd_avg	svrwave	q_svrwave_xauusd_avg_1	00:00:01	00:00:00	EEST	{xauusd_avg_bid}	14 days	f
q_svrwave_xauusd_avg_43200	xauusd_avg	svrwave	q_svrwave_xauusd_avg_43200	12:00:00	00:00:05	EEST	{xauusd_avg_bid}	14 days	f
q_svrwave_test_xauusd_avg_43200	xauusd_avg	svrwave	q_svrwave_xauusd_avg_43200	12:00:00	00:00:05	EEST	{xauusd_avg_bid}	14 days	f
\.


--
-- Data for Name: iq_scaling_factors; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.iq_scaling_factors (id, dataset_id, input_queue_table_name, scaling_factor, dc_offset, input_queue_column_name) FROM stdin;
\.


--
-- Data for Name: models; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.models (id, ensemble_id, decon_level, learning_levels, model_binary, last_modified_time, last_modeled_value_time, norm_mean_coef, norm_range_coef, gradients) FROM stdin;
\.


--
-- Data for Name: multival_requests; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.multival_requests (request_id, dataset_id, user_name, request_time, value_time_start, value_time_end, resolution, value_columns, processed) FROM stdin;
\.


--
-- Data for Name: multival_results; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.multival_results (response_id, request_id, value_time, value_column, value) FROM stdin;
16934	0	2025-07-07 03:00:00	xauusd_avg_bid	3328.764331077383
\.


--
-- Data for Name: parmtune_decrement_tasks; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.parmtune_decrement_tasks (id, dataset_id, start_task_time, end_task_time, start_train_time, end_train_time, start_validation_time, end_validation_time, parameters, status, decrement_step, vp_sliding_direction, vp_slide_count, vp_slide_period_sec, "values", suggested_value) FROM stdin;
\.


--
-- Data for Name: prediction_tasks; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.prediction_tasks (id, dataset_id, start_time, end_time, start_prediction_time, end_prediction_time, status, mse) FROM stdin;
\.


--
-- Data for Name: svr_parameters; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.svr_parameters (id, dataset_id, input_queue_table_name, input_queue_column_name, decon_level, svr_c, svr_epsilon, svr_kernel_param, svr_kernel_param2, svr_decremental_distance, svr_adjacent_levels_ratio, svr_kernel_type, lag_count, levels, step, chunk_ix, grad_level, svr_kernel_param3) FROM stdin;
\.


--
-- Data for Name: user_datasets; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.user_datasets (user_id, dataset_id) FROM stdin;
\.


--
-- Data for Name: userauth; Type: TABLE DATA; Schema: public; Owner: svrwave
--

COPY public.userauth (user_id, username, password, email, name, role, priority) FROM stdin;
100	svrwave	742dd04a2f302afc59f0d83e6d5d5242	svrwave@localhost	svrwave user	USER	0
\.


--
-- Data for Name: w_scaling_factors; Type: TABLE DATA; Schema: public; Owner: asenzz
--

COPY public.w_scaling_factors (id, dataset_id, step, scaling_factor, dc_offset) FROM stdin;
\.


--
-- Name: autotune_tasks_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.autotune_tasks_seq', 100, false);


--
-- Name: datasets_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.datasets_seq', 101, true);


--
-- Name: dq_scaling_factors_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.dq_scaling_factors_seq', 100, false);


--
-- Name: ensembles_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.ensembles_seq', 100, false);


--
-- Name: iq_scaling_factors_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.iq_scaling_factors_seq', 100, false);


--
-- Name: models_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.models_seq', 100, false);


--
-- Name: parmtune_decrement_tasks_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.parmtune_decrement_tasks_seq', 100, false);


--
-- Name: prediction_tasks_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.prediction_tasks_seq', 100, false);


--
-- Name: request_id_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.request_id_seq', 100, false);


--
-- Name: result_id_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.result_id_seq', 17292, true);


--
-- Name: svr_parameters_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.svr_parameters_seq', 100, false);


--
-- Name: userauth_seq; Type: SEQUENCE SET; Schema: public; Owner: svrwave
--

SELECT pg_catalog.setval('public.userauth_seq', 101, true);


--
-- Name: autotune_tasks autotune_tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.autotune_tasks
    ADD CONSTRAINT autotune_tasks_pkey PRIMARY KEY (id);


--
-- Name: datasets datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);


--
-- Name: decon_queue_template decon_queue_template_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.decon_queue_template
    ADD CONSTRAINT decon_queue_template_pkey PRIMARY KEY (value_time);


--
-- Name: decon_queues decon_queues_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.decon_queues
    ADD CONSTRAINT decon_queues_pkey PRIMARY KEY (table_name);


--
-- Name: dq_scaling_factors dq_scaling_factors_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.dq_scaling_factors
    ADD CONSTRAINT dq_scaling_factors_pkey PRIMARY KEY (id);


--
-- Name: ensembles ensembles_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.ensembles
    ADD CONSTRAINT ensembles_pkey PRIMARY KEY (id);


--
-- Name: input_queue_reconciled_hours input_queue_reconciled_hours_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.input_queue_reconciled_hours
    ADD CONSTRAINT input_queue_reconciled_hours_pkey PRIMARY KEY (table_name);


--
-- Name: input_queue_template input_queue_template_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.input_queue_template
    ADD CONSTRAINT input_queue_template_pkey PRIMARY KEY (value_time);


--
-- Name: input_queues input_queues_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.input_queues
    ADD CONSTRAINT input_queues_pkey PRIMARY KEY (table_name);


--
-- Name: iq_scaling_factors iq_scaling_factors_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.iq_scaling_factors
    ADD CONSTRAINT iq_scaling_factors_pkey PRIMARY KEY (id);


--
-- Name: models models_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_pkey PRIMARY KEY (id);


--
-- Name: parmtune_decrement_tasks pk_parmtune_decrement_tasks_id; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.parmtune_decrement_tasks
    ADD CONSTRAINT pk_parmtune_decrement_tasks_id PRIMARY KEY (id);


--
-- Name: multival_requests pk_requests; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.multival_requests
    ADD CONSTRAINT pk_requests PRIMARY KEY (request_id);


--
-- Name: multival_results pk_responses; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.multival_results
    ADD CONSTRAINT pk_responses PRIMARY KEY (response_id);


--
-- Name: prediction_tasks prediction_tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.prediction_tasks
    ADD CONSTRAINT prediction_tasks_pkey PRIMARY KEY (id);


--
-- Name: svr_parameters svr_parameters_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.svr_parameters
    ADD CONSTRAINT svr_parameters_pkey PRIMARY KEY (id);


--
-- Name: decon_queues unique_decon_per_dataset_and_inpu_queue; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.decon_queues
    ADD CONSTRAINT unique_decon_per_dataset_and_inpu_queue UNIQUE (input_queue_table_name, input_queue_column_name, dataset_id);


--
-- Name: svr_parameters unique_svr_parameters; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.svr_parameters
    ADD CONSTRAINT unique_svr_parameters UNIQUE (dataset_id, input_queue_table_name, input_queue_column_name, decon_level);


--
-- Name: userauth unique_user_name; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.userauth
    ADD CONSTRAINT unique_user_name UNIQUE (username);


--
-- Name: user_datasets user_datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.user_datasets
    ADD CONSTRAINT user_datasets_pkey PRIMARY KEY (user_id, dataset_id);


--
-- Name: userauth userauth_pkey; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.userauth
    ADD CONSTRAINT userauth_pkey PRIMARY KEY (user_id);


--
-- Name: w_scaling_factors w_scaling_factors_pkey; Type: CONSTRAINT; Schema: public; Owner: asenzz
--

ALTER TABLE ONLY public.w_scaling_factors
    ADD CONSTRAINT w_scaling_factors_pkey PRIMARY KEY (id);


--
-- Name: fki_multival_requests_dataset_id; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX fki_multival_requests_dataset_id ON public.multival_requests USING btree (dataset_id);


--
-- Name: fki_multival_requests_user_name; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX fki_multival_requests_user_name ON public.multival_requests USING btree (user_name);


--
-- Name: fki_multival_requests_value_time_start; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX fki_multival_requests_value_time_start ON public.multival_requests USING btree (value_time_start, value_time_end, resolution);


--
-- Name: fki_multival_results_request_id; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX fki_multival_results_request_id ON public.multival_results USING btree (request_id);


--
-- Name: fki_multival_results_value_time_start; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX fki_multival_results_value_time_start ON public.multival_results USING btree (value_time, value_column);


--
-- Name: i_input_queue_reconciled_hours; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX i_input_queue_reconciled_hours ON public.input_queue_reconciled_hours USING btree (table_name, start_time DESC, end_time DESC);

ALTER TABLE public.input_queue_reconciled_hours CLUSTER ON i_input_queue_reconciled_hours;


--
-- Name: i_multival_requests_processed; Type: INDEX; Schema: public; Owner: svrwave
--

CREATE INDEX i_multival_requests_processed ON public.multival_requests USING btree (processed);


--
-- Name: autotune_tasks autotune_tasks_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.autotune_tasks
    ADD CONSTRAINT autotune_tasks_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE;


--
-- Name: datasets datasets_main_input_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_main_input_queue_table_name_fkey FOREIGN KEY (main_input_queue_table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE;


--
-- Name: decon_queues decon_queues_input_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.decon_queues
    ADD CONSTRAINT decon_queues_input_queue_table_name_fkey FOREIGN KEY (input_queue_table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE;


--
-- Name: dq_scaling_factors dq_scaling_factors_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.dq_scaling_factors
    ADD CONSTRAINT dq_scaling_factors_dataset_id_fkey FOREIGN KEY (model_id) REFERENCES public.datasets(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: dq_scaling_factors dq_scaling_factors_input_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.dq_scaling_factors
    ADD CONSTRAINT dq_scaling_factors_input_queue_table_name_fkey FOREIGN KEY (input_queue_table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: ensembles ensembles_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.ensembles
    ADD CONSTRAINT ensembles_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE;


--
-- Name: ensembles ensembles_decon_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.ensembles
    ADD CONSTRAINT ensembles_decon_queue_table_name_fkey FOREIGN KEY (decon_queue_table_name) REFERENCES public.decon_queues(table_name) ON UPDATE CASCADE;


--
-- Name: input_queue_reconciled_hours fk_input_queues_table_name; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.input_queue_reconciled_hours
    ADD CONSTRAINT fk_input_queues_table_name FOREIGN KEY (table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: multival_requests fk_multival_requests_dataset_id; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.multival_requests
    ADD CONSTRAINT fk_multival_requests_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.datasets(id);


--
-- Name: multival_requests fk_multival_requests_user_name; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.multival_requests
    ADD CONSTRAINT fk_multival_requests_user_name FOREIGN KEY (user_name) REFERENCES public.userauth(username);


--
-- Name: parmtune_decrement_tasks fk_parmtune_decrement_dataset_id; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.parmtune_decrement_tasks
    ADD CONSTRAINT fk_parmtune_decrement_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.datasets(id);


--
-- Name: input_queues input_queues_user_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.input_queues
    ADD CONSTRAINT input_queues_user_name_fkey FOREIGN KEY (user_name) REFERENCES public.userauth(username);


--
-- Name: iq_scaling_factors iq_scaling_factors_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.iq_scaling_factors
    ADD CONSTRAINT iq_scaling_factors_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: iq_scaling_factors iq_scaling_factors_input_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.iq_scaling_factors
    ADD CONSTRAINT iq_scaling_factors_input_queue_table_name_fkey FOREIGN KEY (input_queue_table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: models models_ensemble_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_ensemble_id_fkey FOREIGN KEY (ensemble_id) REFERENCES public.ensembles(id) ON UPDATE CASCADE;


--
-- Name: prediction_tasks prediction_tasks_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.prediction_tasks
    ADD CONSTRAINT prediction_tasks_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE;


--
-- Name: svr_parameters svr_parameters_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.svr_parameters
    ADD CONSTRAINT svr_parameters_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE;


--
-- Name: svr_parameters svr_parameters_input_queue_table_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.svr_parameters
    ADD CONSTRAINT svr_parameters_input_queue_table_name_fkey FOREIGN KEY (input_queue_table_name) REFERENCES public.input_queues(table_name) ON UPDATE CASCADE;


--
-- Name: user_datasets user_datasets_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.user_datasets
    ADD CONSTRAINT user_datasets_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: user_datasets user_datasets_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.user_datasets
    ADD CONSTRAINT user_datasets_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.userauth(user_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO svrwave;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- Name: TABLE w_scaling_factors; Type: ACL; Schema: public; Owner: asenzz
--

GRANT ALL ON TABLE public.w_scaling_factors TO svrwave;


--
-- PostgreSQL database dump complete
--

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.1
-- Dumped by pg_dump version 16.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: q_svrwave_xauusd_avg_1; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.q_svrwave_xauusd_avg_1 (
    xauusd_avg_bid double precision DEFAULT 0 NOT NULL,
    xauusd_avg_ask double precision DEFAULT 0
)
INHERITS (public.input_queue_template);


ALTER TABLE public.q_svrwave_xauusd_avg_1 OWNER TO svrwave;

--
-- Name: q_svrwave_test_xauusd_avg_1; Type: VIEW; Schema: public; Owner: svrwave
--

CREATE VIEW public.q_svrwave_test_xauusd_avg_1 AS
 SELECT value_time,
    update_time,
    tick_volume,
    xauusd_avg_bid,
    xauusd_avg_ask
   FROM ( SELECT q_svrwave_xauusd_avg_1.value_time,
            q_svrwave_xauusd_avg_1.update_time,
            q_svrwave_xauusd_avg_1.tick_volume,
            q_svrwave_xauusd_avg_1.xauusd_avg_bid,
            q_svrwave_xauusd_avg_1.xauusd_avg_ask
           FROM public.q_svrwave_xauusd_avg_1
          WHERE (q_svrwave_xauusd_avg_1.value_time < '2025-07-21 22:29:58'::timestamp without time zone)
          ORDER BY q_svrwave_xauusd_avg_1.value_time DESC
         LIMIT (2524 * 43200)) unnamed_subquery
  ORDER BY value_time;


ALTER VIEW public.q_svrwave_test_xauusd_avg_1 OWNER TO svrwave;

--
-- Name: q_svrwave_xauusd_avg_3600; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.q_svrwave_xauusd_avg_3600 (
    xauusd_avg_bid double precision DEFAULT 0 NOT NULL,
    xauusd_avg_ask double precision DEFAULT 0 NOT NULL
)
INHERITS (public.input_queue_template);


ALTER TABLE public.q_svrwave_xauusd_avg_3600 OWNER TO svrwave;

--
-- Name: q_svrwave_test_xauusd_avg_3600; Type: VIEW; Schema: public; Owner: svrwave
--

CREATE VIEW public.q_svrwave_test_xauusd_avg_3600 AS
 SELECT value_time,
    update_time,
    tick_volume,
    xauusd_avg_bid,
    xauusd_avg_ask
   FROM ( SELECT q_svrwave_xauusd_avg_3600.value_time,
            q_svrwave_xauusd_avg_3600.update_time,
            q_svrwave_xauusd_avg_3600.tick_volume,
            q_svrwave_xauusd_avg_3600.xauusd_avg_bid,
            q_svrwave_xauusd_avg_3600.xauusd_avg_ask
           FROM public.q_svrwave_xauusd_avg_3600
          WHERE (q_svrwave_xauusd_avg_3600.value_time < '2025-07-21 22:29:58'::timestamp without time zone)
          ORDER BY q_svrwave_xauusd_avg_3600.value_time DESC
         LIMIT 5678) unnamed_subquery
  ORDER BY value_time;


ALTER VIEW public.q_svrwave_test_xauusd_avg_3600 OWNER TO svrwave;

--
-- Name: q_svrwave_xauusd_avg_43200; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.q_svrwave_xauusd_avg_43200 (
    value_time timestamp without time zone,
    update_time timestamp without time zone,
    tick_volume double precision,
    xauusd_avg_bid double precision,
    xauusd_avg_ask double precision
);


ALTER TABLE public.q_svrwave_xauusd_avg_43200 OWNER TO svrwave;

--
-- Name: q_svrwave_test_xauusd_avg_43200; Type: VIEW; Schema: public; Owner: svrwave
--

CREATE VIEW public.q_svrwave_test_xauusd_avg_43200 AS
 SELECT value_time,
    update_time,
    tick_volume,
    xauusd_avg_bid,
    xauusd_avg_ask
   FROM ( SELECT q_svrwave_xauusd_avg_43200.value_time,
            q_svrwave_xauusd_avg_43200.update_time,
            q_svrwave_xauusd_avg_43200.tick_volume,
            q_svrwave_xauusd_avg_43200.xauusd_avg_bid,
            q_svrwave_xauusd_avg_43200.xauusd_avg_ask
           FROM public.q_svrwave_xauusd_avg_43200
          WHERE (q_svrwave_xauusd_avg_43200.value_time < '2025-07-21 22:29:58'::timestamp without time zone)
          ORDER BY q_svrwave_xauusd_avg_43200.value_time DESC
         LIMIT 2524) unnamed_subquery
  ORDER BY value_time;


ALTER VIEW public.q_svrwave_test_xauusd_avg_43200 OWNER TO svrwave;

--
-- Name: z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid (
    level_0 double precision DEFAULT 0 NOT NULL,
    level_1 double precision DEFAULT 0 NOT NULL,
    level_2 double precision DEFAULT 0 NOT NULL,
    level_3 double precision DEFAULT 0 NOT NULL,
    level_4 double precision DEFAULT 0 NOT NULL,
    level_5 double precision DEFAULT 0 NOT NULL,
    level_6 double precision DEFAULT 0 NOT NULL,
    level_7 double precision DEFAULT 0 NOT NULL,
    level_8 double precision DEFAULT 0 NOT NULL,
    level_9 double precision DEFAULT 0 NOT NULL,
    level_10 double precision DEFAULT 0 NOT NULL,
    level_11 double precision DEFAULT 0 NOT NULL,
    level_12 double precision DEFAULT 0 NOT NULL,
    level_13 double precision DEFAULT 0 NOT NULL,
    level_14 double precision DEFAULT 0 NOT NULL,
    level_15 double precision DEFAULT 0 NOT NULL,
    level_16 double precision DEFAULT 0 NOT NULL,
    level_17 double precision DEFAULT 0 NOT NULL,
    level_18 double precision DEFAULT 0 NOT NULL,
    level_19 double precision DEFAULT 0 NOT NULL,
    level_20 double precision DEFAULT 0 NOT NULL,
    level_21 double precision DEFAULT 0 NOT NULL,
    level_22 double precision DEFAULT 0 NOT NULL,
    level_23 double precision DEFAULT 0 NOT NULL,
    level_24 double precision DEFAULT 0 NOT NULL,
    level_25 double precision DEFAULT 0 NOT NULL,
    level_26 double precision DEFAULT 0 NOT NULL,
    level_27 double precision DEFAULT 0 NOT NULL,
    level_28 double precision DEFAULT 0 NOT NULL,
    level_29 double precision DEFAULT 0 NOT NULL,
    level_30 double precision DEFAULT 0 NOT NULL,
    level_31 double precision DEFAULT 0 NOT NULL,
    level_32 double precision DEFAULT 0 NOT NULL,
    level_33 double precision DEFAULT 0 NOT NULL,
    level_34 double precision DEFAULT 0 NOT NULL,
    level_35 double precision DEFAULT 0 NOT NULL,
    level_36 double precision DEFAULT 0 NOT NULL,
    level_37 double precision DEFAULT 0 NOT NULL,
    level_38 double precision DEFAULT 0 NOT NULL,
    level_39 double precision DEFAULT 0 NOT NULL,
    level_40 double precision DEFAULT 0 NOT NULL,
    level_41 double precision DEFAULT 0 NOT NULL,
    level_42 double precision DEFAULT 0 NOT NULL,
    level_43 double precision DEFAULT 0 NOT NULL,
    level_44 double precision DEFAULT 0 NOT NULL,
    level_45 double precision DEFAULT 0 NOT NULL,
    level_46 double precision DEFAULT 0 NOT NULL,
    level_47 double precision DEFAULT 0 NOT NULL,
    level_48 double precision DEFAULT 0 NOT NULL,
    level_49 double precision DEFAULT 0 NOT NULL,
    level_50 double precision DEFAULT 0 NOT NULL,
    level_51 double precision DEFAULT 0 NOT NULL,
    level_52 double precision DEFAULT 0 NOT NULL,
    level_53 double precision DEFAULT 0 NOT NULL,
    level_54 double precision DEFAULT 0 NOT NULL,
    level_55 double precision DEFAULT 0 NOT NULL,
    level_56 double precision DEFAULT 0 NOT NULL,
    level_57 double precision DEFAULT 0 NOT NULL,
    level_58 double precision DEFAULT 0 NOT NULL,
    level_59 double precision DEFAULT 0 NOT NULL,
    level_60 double precision DEFAULT 0 NOT NULL,
    level_61 double precision DEFAULT 0 NOT NULL,
    level_62 double precision DEFAULT 0 NOT NULL,
    level_63 double precision DEFAULT 0 NOT NULL
)
INHERITS (public.decon_queue_template);


ALTER TABLE public.z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid OWNER TO svrwave;

--
-- Name: z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid; Type: TABLE; Schema: public; Owner: svrwave
--

CREATE TABLE public.z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid (
    level_0 double precision DEFAULT 0 NOT NULL,
    level_1 double precision DEFAULT 0 NOT NULL,
    level_2 double precision DEFAULT 0 NOT NULL,
    level_3 double precision DEFAULT 0 NOT NULL,
    level_4 double precision DEFAULT 0 NOT NULL,
    level_5 double precision DEFAULT 0 NOT NULL,
    level_6 double precision DEFAULT 0 NOT NULL,
    level_7 double precision DEFAULT 0 NOT NULL,
    level_8 double precision DEFAULT 0 NOT NULL,
    level_9 double precision DEFAULT 0 NOT NULL,
    level_10 double precision DEFAULT 0 NOT NULL,
    level_11 double precision DEFAULT 0 NOT NULL,
    level_12 double precision DEFAULT 0 NOT NULL,
    level_13 double precision DEFAULT 0 NOT NULL,
    level_14 double precision DEFAULT 0 NOT NULL,
    level_15 double precision DEFAULT 0 NOT NULL,
    level_16 double precision DEFAULT 0 NOT NULL,
    level_17 double precision DEFAULT 0 NOT NULL,
    level_18 double precision DEFAULT 0 NOT NULL,
    level_19 double precision DEFAULT 0 NOT NULL,
    level_20 double precision DEFAULT 0 NOT NULL,
    level_21 double precision DEFAULT 0 NOT NULL,
    level_22 double precision DEFAULT 0 NOT NULL,
    level_23 double precision DEFAULT 0 NOT NULL,
    level_24 double precision DEFAULT 0 NOT NULL,
    level_25 double precision DEFAULT 0 NOT NULL,
    level_26 double precision DEFAULT 0 NOT NULL,
    level_27 double precision DEFAULT 0 NOT NULL,
    level_28 double precision DEFAULT 0 NOT NULL,
    level_29 double precision DEFAULT 0 NOT NULL,
    level_30 double precision DEFAULT 0 NOT NULL,
    level_31 double precision DEFAULT 0 NOT NULL,
    level_32 double precision DEFAULT 0 NOT NULL,
    level_33 double precision DEFAULT 0 NOT NULL,
    level_34 double precision DEFAULT 0 NOT NULL,
    level_35 double precision DEFAULT 0 NOT NULL,
    level_36 double precision DEFAULT 0 NOT NULL,
    level_37 double precision DEFAULT 0 NOT NULL,
    level_38 double precision DEFAULT 0 NOT NULL,
    level_39 double precision DEFAULT 0 NOT NULL,
    level_40 double precision DEFAULT 0 NOT NULL,
    level_41 double precision DEFAULT 0 NOT NULL,
    level_42 double precision DEFAULT 0 NOT NULL,
    level_43 double precision DEFAULT 0 NOT NULL,
    level_44 double precision DEFAULT 0 NOT NULL,
    level_45 double precision DEFAULT 0 NOT NULL,
    level_46 double precision DEFAULT 0 NOT NULL,
    level_47 double precision DEFAULT 0 NOT NULL,
    level_48 double precision DEFAULT 0 NOT NULL,
    level_49 double precision DEFAULT 0 NOT NULL,
    level_50 double precision DEFAULT 0 NOT NULL,
    level_51 double precision DEFAULT 0 NOT NULL,
    level_52 double precision DEFAULT 0 NOT NULL,
    level_53 double precision DEFAULT 0 NOT NULL,
    level_54 double precision DEFAULT 0 NOT NULL,
    level_55 double precision DEFAULT 0 NOT NULL,
    level_56 double precision DEFAULT 0 NOT NULL,
    level_57 double precision DEFAULT 0 NOT NULL,
    level_58 double precision DEFAULT 0 NOT NULL,
    level_59 double precision DEFAULT 0 NOT NULL,
    level_60 double precision DEFAULT 0 NOT NULL,
    level_61 double precision DEFAULT 0 NOT NULL,
    level_62 double precision DEFAULT 0 NOT NULL,
    level_63 double precision DEFAULT 0 NOT NULL
)
INHERITS (public.decon_queue_template);


ALTER TABLE public.z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid OWNER TO svrwave;

--
-- Name: q_svrwave_xauusd_avg_1 update_time; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_1 ALTER COLUMN update_time SET DEFAULT CURRENT_TIMESTAMP;


--
-- Name: q_svrwave_xauusd_avg_1 tick_volume; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_1 ALTER COLUMN tick_volume SET DEFAULT 1;


--
-- Name: q_svrwave_xauusd_avg_3600 update_time; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_3600 ALTER COLUMN update_time SET DEFAULT CURRENT_TIMESTAMP;


--
-- Name: q_svrwave_xauusd_avg_3600 tick_volume; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_3600 ALTER COLUMN tick_volume SET DEFAULT 1;


--
-- Name: z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid update_time; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid ALTER COLUMN update_time SET DEFAULT now();


--
-- Name: z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid tick_volume; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid ALTER COLUMN tick_volume SET DEFAULT 1.0;


--
-- Name: z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid update_time; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid ALTER COLUMN update_time SET DEFAULT now();


--
-- Name: z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid tick_volume; Type: DEFAULT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid ALTER COLUMN tick_volume SET DEFAULT 1.0;


--
-- Name: q_svrwave_xauusd_avg_1 q_svrwave_xauusd_avg_1_pk; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_1
    ADD CONSTRAINT q_svrwave_xauusd_avg_1_pk PRIMARY KEY (value_time);


--
-- Name: q_svrwave_xauusd_avg_3600 q_svrwave_xauusd_avg_3600_pk; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.q_svrwave_xauusd_avg_3600
    ADD CONSTRAINT q_svrwave_xauusd_avg_3600_pk PRIMARY KEY (value_time);


--
-- Name: z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid_pk; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid
    ADD CONSTRAINT z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid_pk PRIMARY KEY (value_time);


--
-- Name: z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid_pk; Type: CONSTRAINT; Schema: public; Owner: svrwave
--

ALTER TABLE ONLY public.z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid
    ADD CONSTRAINT z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid_pk PRIMARY KEY (value_time);


--
-- PostgreSQL database dump complete
--
