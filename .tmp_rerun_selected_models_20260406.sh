#!/usr/bin/env bash
set -euo pipefail

cd /home/yiming/research/CWF

export REPO_ROOT="/home/yiming/research/CWF"
export SAMPLER_EXE="/home/yiming/research/CWFSampling/build/vcg_poisson_sampling"
export CWF_EXE="${REPO_ROOT}/bin/cwf"
export QUADCOVER_EXE="${REPO_ROOT}/bin/quadcover_main"
export SAMPLE_NUM="30000"
export SAMPLE_SEED="0"
export CWF_ITERS="50"
export QUADCOVER_THREADS="16"
export PIPELINE_SUBDIR="rerun_cwf_quadcover_n30000"

cleanup_cwf_dir() {
  local cwf_dir="$1"
  local keep_rvd="$2"
  local keep_remesh="$3"
  local keep_log="$4"
  find "${cwf_dir}" -maxdepth 1 -type f \
    ! -name "$(basename "${keep_rvd}")" \
    ! -name "$(basename "${keep_remesh}")" \
    ! -name "$(basename "${keep_log}")" \
    -delete
}

cleanup_quadcover_dir() {
  local quadcover_dir="$1"
  local keep_remesh="$2"
  local keep_csv="$3"
  local keep_log="$4"
  find "${quadcover_dir}" -maxdepth 1 -type f \
    ! -name "$(basename "${keep_remesh}")" \
    ! -name "$(basename "${keep_csv}")" \
    ! -name "$(basename "${keep_log}")" \
    -delete
}

run_one() {
  local model="$1"
  local work_dir="${REPO_ROOT}/quadResult/.work/${model}"
  local normalized_mesh="${work_dir}/${model}_normalized.obj"
  local sample_points="${work_dir}/n${SAMPLE_NUM}_${model}_inputPoints.xyz"
  local pipeline_dir="${REPO_ROOT}/quadResult/${model}/${PIPELINE_SUBDIR}"
  local sampler_log="${pipeline_dir}/sampling_rerun.log"
  local cwf_dir="${pipeline_dir}/cwf"
  local cwf_log="${cwf_dir}/cwf_rerun.log"
  local quadcover_dir="${pipeline_dir}/quadcover"
  local quadcover_log="${quadcover_dir}/quadcover_rerun.log"
  local cwf_prefix="CWF${CWF_ITERS}_${model}"
  local standardized_rvd="${cwf_dir}/${cwf_prefix}_RVD.obj"
  local standardized_remesh="${cwf_dir}/${cwf_prefix}_Remesh.obj"
  local final_quadcover_remesh="${quadcover_dir}/QuadCoverRerun_${model}_Remesh.obj"
  local final_quadcover_csv="${quadcover_dir}/QuadCoverRerun_${model}_Spheres.csv"
  local raw_rvd=""
  local raw_remesh=""
  local raw_quadcover_remesh=""
  local raw_quadcover_csv=""

  echo "[launch] ${model} $(date '+%F %T')"

  if [[ ! -f "${normalized_mesh}" ]]; then
    echo "[error] missing normalized mesh: ${normalized_mesh}" >&2
    return 1
  fi

  rm -rf "${pipeline_dir}"
  mkdir -p "${cwf_dir}" "${quadcover_dir}"

  {
    echo "[Rerun] model=${model}"
    echo "[Rerun] normalized_mesh=${normalized_mesh}"
    echo "[Rerun] sample_points=${sample_points}"
    echo "[Rerun] sample_num=${SAMPLE_NUM}"
    echo "[Rerun] seed=${SAMPLE_SEED}"
    echo "[Rerun] command=${SAMPLER_EXE} ${normalized_mesh} ${sample_points} ${SAMPLE_NUM} ${SAMPLE_SEED}"
    echo
    "${SAMPLER_EXE}" "${normalized_mesh}" "${sample_points}" "${SAMPLE_NUM}" "${SAMPLE_SEED}"
  } > "${sampler_log}" 2>&1

  {
    echo "[Rerun] model=${model}"
    echo "[Rerun] normalized_mesh=${normalized_mesh}"
    echo "[Rerun] sample_points=${sample_points}"
    echo "[Rerun] cwf_iterations=${CWF_ITERS}"
    echo "[Rerun] command=${CWF_EXE} ${normalized_mesh} ${sample_points} ${CWF_ITERS}"
    echo
    (
      cd "${cwf_dir}"
      "${CWF_EXE}" "${normalized_mesh}" "${sample_points}" "${CWF_ITERS}"
    )
  } > "${cwf_log}" 2>&1

  raw_rvd="$(find "${cwf_dir}" -maxdepth 1 -type f -name '*_RVD.obj' ! -name '*_Iter*' | sort | tail -n 1)"
  raw_remesh="$(find "${cwf_dir}" -maxdepth 1 -type f -name '*Remesh.obj' ! -name '*_Iter*' | sort | tail -n 1)"
  if [[ -z "${raw_rvd}" || -z "${raw_remesh}" ]]; then
    echo "[error] CWF finished but final outputs were not found for ${model}" >&2
    return 1
  fi

  cp "${raw_rvd}" "${standardized_rvd}"
  cp "${raw_remesh}" "${standardized_remesh}"
  cleanup_cwf_dir "${cwf_dir}" "${standardized_rvd}" "${standardized_remesh}" "${cwf_log}"

  {
    echo "[Rerun] model=${model}"
    echo "[Rerun] normalized_mesh=${normalized_mesh}"
    echo "[Rerun] init_remesh=${standardized_remesh}"
    echo "[Rerun] quadcover_threads=${QUADCOVER_THREADS}"
    echo "[Rerun] command=${QUADCOVER_EXE} --surface ${normalized_mesh} --input ${standardized_remesh} --name ${model} --output ${quadcover_dir} --threads ${QUADCOVER_THREADS} --final-only"
    echo
    (
      cd "${quadcover_dir}"
      "${QUADCOVER_EXE}" \
        --surface "${normalized_mesh}" \
        --input "${standardized_remesh}" \
        --name "${model}" \
        --output "${quadcover_dir}" \
        --threads "${QUADCOVER_THREADS}" \
        --final-only
    )
  } > "${quadcover_log}" 2>&1

  raw_quadcover_remesh="$(find "${quadcover_dir}" -maxdepth 1 -type f -name '*_Remesh.obj' ! -name '*_Iter*' ! -name 'QuadCoverRerun_*' | sort | tail -n 1)"
  raw_quadcover_csv="$(find "${quadcover_dir}" -maxdepth 1 -type f -name '*_Spheres.csv' ! -name '*_Iter*' ! -name 'QuadCoverRerun_*' | sort | tail -n 1)"
  if [[ -z "${raw_quadcover_remesh}" || -z "${raw_quadcover_csv}" ]]; then
    echo "[error] QuadCover finished but final outputs were not found for ${model}" >&2
    return 1
  fi

  cp "${raw_quadcover_remesh}" "${final_quadcover_remesh}"
  cp "${raw_quadcover_csv}" "${final_quadcover_csv}"
  cleanup_quadcover_dir "${quadcover_dir}" "${final_quadcover_remesh}" "${final_quadcover_csv}" "${quadcover_log}"

  echo "[done] ${model} $(date '+%F %T')"
}

export -f run_one

printf '%s\n' \
  '11_open_end_rod' \
  '16_c_shaped_arc_segment' \
  '06_angled_fork_bracket' \
  '06_cylindrical_block_with_v_notch' \
  '05_arched_chute_bracket' \
  '19_spur_gear_ring' \
  '18_framed_anchor_bracket' \
  '19_socket_head_bolt' \
| xargs -I{} -P8 bash -lc 'run_one "$@"' _ {}
