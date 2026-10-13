"use strict";

const data = window.MUTA_DATA;
const repo = "https://github.com/kevinhyj/MutaHarness/blob/main/";
const experiments = {
  zero: {
    methods: [
      { key: "basic", label: "基础原生Codex", short: "基础" },
      { key: "msa_tools", label: "仅MSA", short: "MSA" },
      { key: "structure_tools", label: "仅结构", short: "结构" },
      { key: "joint_tools", label: "Structure + MSA", short: "联合" },
    ],
    control: "basic",
    improved: "joint_tools",
    comparison: "联合 − 基础",
    caption: "MSA、结构列分别对应仅MSA、仅结构。",
  },
  ensemble: {
    methods: [
      { key: "direct", label: "单次原生Codex", short: "单次" },
      { key: "cv", label: "同预算CV单程序", short: "CV单程序" },
      { key: "ensemble", label: "三程序排名融合", short: "集成" },
    ],
    control: "cv",
    improved: "ensemble",
    comparison: "集成 − 同预算CV单程序",
    caption: "CV指用已测反馈做交叉验证，在三份程序中选一个。",
  },
};
const state = {
  mode: "zero",
  // The two experiments retain separate selections because their candidate sets differ.
  selected: Object.fromEntries(
    Object.keys(experiments).map((mode) => [mode, data[mode].cases[0].id]),
  ),
};
const element = (id) => document.getElementById(id);
const formatScore = (value) => value.toFixed(4);
const formatDelta = (value) => (value >= 0 ? "+" : "") + formatScore(value);
const tone = (value) => (value >= 0 ? "positive" : "negative");
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[char],
  );
const current = () => experiments[state.mode];
const delta = (scores) =>
  scores[current().improved] - scores[current().control];
const isBest = (value, scores) =>
  Math.abs(value - Math.max(...Object.values(scores))) < 1e-12;

function scoreCells(scores) {
  return current()
    .methods.map(({ key }) => {
      const value = formatScore(scores[key]);
      return `<td data-method="${key}">${isBest(scores[key], scores) ? `<strong>${value}</strong>` : value}</td>`;
    })
    .join("");
}

function scoreCards(scores) {
  return current()
    .methods.map(
      ({ key, label }) => `
    <div class="metric ${isBest(scores[key], scores) ? "best" : ""}">
      <span>${label}</span><span class="value" data-method="${key}">${formatScore(scores[key])}</span>
    </div>`,
    )
    .join("");
}

function renderExplanation(case_) {
  element("detail-title").textContent = `${case_.label} · 结果说明`;
  element("source-link").href = repo + case_.source;
  const explanation = element("explanation");
  explanation.replaceChildren(
    element(`${state.mode}-detail`).content.cloneNode(true),
  );
  if (state.mode === "zero") return;

  explanation.querySelector("[data-cv-program]").textContent =
    `程序 ${case_.cv_selected + 1}`;
  explanation.querySelector("[data-program-links]").innerHTML = [0, 1, 2]
    .map(
      (index) =>
        `<a href="${repo}programs/${escapeHtml(case_.id)}/candidate_0${index}/proposal.py">程序 ${index + 1} ↗</a>`,
    )
    .join("");
  const candidates = [...case_.candidates].sort(
    (a, b) => b.score - a.score || a.id.localeCompare(b.id),
  );
  explanation.querySelector("[data-candidates]").innerHTML = candidates
    .slice(0, 10)
    .map(
      (candidate) => `
    <tr><td>${escapeHtml(candidate.mutation)}</td><td>${candidate.score.toFixed(3)}</td>
    <td>${candidate.disagreement.toFixed(3)}</td></tr>`,
    )
    .join("");
}

function renderSelection() {
  const group = data[state.mode];
  const case_ = group.cases.find(
    (case_) => case_.id === state.selected[state.mode],
  );
  element("case-select").value = case_.id;
  element("case-title").textContent = `当前任务：${case_.label}`;
  element("current-meta").textContent =
    `以下分数仅对应 ${case_.label}；${group.feedback_count}条已测反馈，${group.query_count}个候选。`;
  element("case-scores").innerHTML = scoreCards(case_.scores);
  element("case-delta").innerHTML =
    `${current().comparison}：<strong class="${tone(delta(case_.scores))}">${formatDelta(delta(case_.scores))}</strong>`;
  element("tablebody")
    .querySelectorAll("tr")
    .forEach((row) => {
      const selected = row.dataset.case === case_.id;
      row.classList.toggle("selected", selected);
      row
        .querySelector("button")
        .setAttribute("aria-pressed", String(selected));
    });
  renderExplanation(case_);
}

function renderExperiment() {
  const group = data[state.mode];
  const config = current();
  Object.keys(experiments).forEach((mode) =>
    element(mode).setAttribute("aria-pressed", String(mode === state.mode)),
  );
  document
    .querySelectorAll(".metrics")
    .forEach(
      (cards) =>
        (cards.style.gridTemplateColumns = `repeat(${config.methods.length},minmax(0,1fr))`),
    );
  element("scope").textContent =
    `${data.model} · 每任务${group.feedback_count}条已测反馈、${group.query_count}个候选 · 两项实验分别评估`;
  element("means").innerHTML = scoreCards(group.means);
  const wins = group.cases.filter((case_) => delta(case_.scores) > 0).length;
  const note =
    state.mode === "zero"
      ? ` Top-25平均重合率：基础 ${group.top25_means.basic.toFixed(3)}，联合 ${group.top25_means.joint_tools.toFixed(3)}，未同步改善。`
      : " CV单程序和集成使用同一组三份程序，生成预算相同；原容量失败及交付恢复见完整记录。";
  element("summary-note").textContent =
    `${config.comparison}：平均 ${formatDelta(delta(group.means))}，${wins}/${group.cases.length}个任务提升。` +
    note;
  element("case-select").innerHTML = group.cases
    .map(
      (case_) =>
        `<option value="${escapeHtml(case_.id)}">${escapeHtml(case_.label)}</option>`,
    )
    .join("");
  element("tablehead").innerHTML =
    `<tr><th>任务</th>${config.methods.map(({ short }) => `<th>${short}</th>`).join("")}<th>Δ</th></tr>`;
  element("tablebody").innerHTML = group.cases
    .map(
      (case_) => `
    <tr data-case="${escapeHtml(case_.id)}">
      <td><button aria-pressed="false" aria-label="查看 ${escapeHtml(case_.label)}">${escapeHtml(case_.label)}</button></td>
      ${scoreCells(case_.scores)}<td class="${tone(delta(case_.scores))}">${formatDelta(delta(case_.scores))}</td>
    </tr>`,
    )
    .join("");
  element("tablefoot").innerHTML =
    `<tr><td><strong>10任务平均</strong></td>${scoreCells(group.means)}<td class="${tone(delta(group.means))}"><strong>${formatDelta(delta(group.means))}</strong></td></tr>`;
  element("table-caption").textContent =
    `Δ = ${config.comparison}。${config.caption} 合并10个任务为描述性汇总。`;
  renderSelection();
}

function chooseCase(id) {
  state.selected[state.mode] = id;
  renderSelection();
}

// Delegate row clicks so selection updates keep the focused button in place.
element("tablebody").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-case]");
  if (row) chooseCase(row.dataset.case);
});
element("case-select").addEventListener("change", (event) =>
  chooseCase(event.target.value),
);
Object.keys(experiments).forEach((mode) =>
  element(mode).addEventListener("click", () => {
    state.mode = mode;
    renderExperiment();
  }),
);
renderExperiment();
