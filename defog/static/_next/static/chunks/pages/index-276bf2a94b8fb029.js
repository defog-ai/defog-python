(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{75557:function(e,t,i){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return i(44369)}])},4278:function(e,t,i){"use strict";var s=i(85893),r=i(9008),n=i.n(r);let l=()=>(0,s.jsxs)(n(),{children:[(0,s.jsx)("title",{children:"Defog.ai - AI Assistant for Data Analysis"}),(0,s.jsx)("meta",{name:"description",content:"Train your AI data assistant on your own device"}),(0,s.jsx)("meta",{name:"viewport",content:"width=device-width, initial-scale=1"}),(0,s.jsx)("link",{rel:"icon",href:"/favicon.ico"})]});t.Z=l},35175:function(e,t,i){"use strict";var s=i(85893);i(67294);var r=i(69215);let n=e=>{let{id:t,children:i}=e,{Content:n,Sider:l}=r.Ar,o=[{key:"select-model",title:"Select Model",icon:(0,s.jsx)("a",{href:"/",children:"1. Select Model"})},{key:"extract-metadata",title:"Extract Metadata",icon:(0,s.jsx)("a",{href:"/extract-metadata",children:"2. Extract Metadata"})},{key:"instruct-model",title:"Instruct Model",icon:(0,s.jsx)("a",{href:"/instruct-model",children:"3. Instruct Model"})},{key:"query-database",title:"Query your database",icon:(0,s.jsx)("a",{href:"/query-database",children:"4. Query Database"})}];return(0,s.jsx)(r.Ar,{style:{height:"100vh"},children:(0,s.jsxs)(n,{style:{padding:"50 50"},children:[(0,s.jsx)(l,{style:{height:"100vh",position:"fixed"},children:(0,s.jsx)(r.v2,{style:{width:200,paddingTop:"2em",paddingBottom:"2em"},mode:"inline",selectedKeys:[t],items:o})}),(0,s.jsx)("div",{style:{paddingLeft:240,paddingTop:30},children:i})]})})};t.Z=n},44369:function(e,t,i){"use strict";i.r(t);var s=i(85893),r=i(4278),n=i(35175),l=i(69215);let o=()=>(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(r.Z,{}),(0,s.jsxs)(n.Z,{id:"select-model",children:[(0,s.jsx)("h1",{style:{paddingBottom:"1em"},children:"Select Model"}),(0,s.jsxs)(l.X2,{type:"flex",children:[(0,s.jsx)(l.JX,{md:{span:8},xs:{span:24},style:{display:"flex"},children:(0,s.jsxs)(l.Zb,{title:"Locally Hosted (Community)",bordered:!0,style:{width:"100%",marginRight:10},children:["\uD83E\uDDBE Model Type: ",(0,s.jsx)("code",{children:"SQLCoder-7b-4_k.GGUF"})," ",(0,s.jsx)(l.u,{title:"Our fastest model with 78% accuracy on `sql-eval`. Works great on Apple Silicon.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83C\uDD93 Free forever! ",(0,s.jsx)(l.u,{title:"The model is free forever.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83E\uDD37\uD83C\uDFFD‍♂️ Not great at following-instructions ",(0,s.jsx)(l.u,{title:"The model can follow basic instructions, but is not great at following specialized ones",children:"ℹ"})," ",(0,s.jsx)("br",{}),"❌ No fine tuning ",(0,s.jsx)(l.u,{title:"Model works great for simple questions that do not require specialized domain knowledge.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDC77\uD83C\uDFFD‍♂️ Limited agent access ",(0,s.jsx)(l.u,{title:"The model is very limited at solving highly complex problems requiring multiple steps.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDD10 Complete data privacy with no data sharing",(0,s.jsx)("hr",{style:{marginTop:"1em",border:"1px solid rgba(0,0,0,0.1)"}}),(0,s.jsxs)("div",{style:{paddingBottom:"2em"},children:[(0,s.jsx)("h3",{children:"Pricing"}),"\uD83C\uDD93 Free forever!"]}),(0,s.jsx)(l.zx,{type:"primary",ghost:!0,style:{position:"absolute",width:"85%",bottom:10,maxWidth:400},children:"Get Started"})]})}),(0,s.jsx)(l.JX,{md:{span:8},xs:{span:24},style:{display:"flex"},children:(0,s.jsxs)(l.Zb,{title:"API Based",bordered:!0,style:{width:"100%",marginRight:10},children:["\uD83E\uDDBE Model Type: ",(0,s.jsx)("code",{children:"SQLCoder-34b-instruct"})," ",(0,s.jsx)(l.u,{title:"Our most capable closed-source model with 91% accuracy on `sql-eval`",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDE80 Usage-based pricing using API credits ",(0,s.jsx)(l.u,{title:"The model is hosted on our servers and can be accessed via API, using a credit based system",children:"ℹ"})," ",(0,s.jsx)("br",{}),"✅ Follows-instructions ",(0,s.jsx)(l.u,{title:"The model is great at following specialized instructions",children:"ℹ"})," ",(0,s.jsx)("br",{}),"❌ No fine tuning ",(0,s.jsx)(l.u,{title:"Model works great for complex questions that do not require specialized domain knowledge",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDC77\uD83C\uDFFD‍♂️ Generalist agent capabilities ",(0,s.jsx)(l.u,{title:"The model is proficient at solving generalist problems involving multiple steps.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDD10 Metadata shared with our SOC-2 compliant server",(0,s.jsx)("hr",{style:{marginTop:"1em",border:"1px solid rgba(0,0,0,0.1)"}}),(0,s.jsxs)("div",{style:{paddingBottom:"2em"},children:[(0,s.jsx)("h3",{children:"Pricing"}),"\uD83C\uDD93 1000 free API credits per month ",(0,s.jsx)("br",{}),"\uD83D\uDCB0 $0.03 per API credit ",(0,s.jsx)("br",{}),"- Every 500 tokens of generated SQL = 1 API credit ",(0,s.jsx)("br",{}),"- Every action taken by an agent = 1 API credit ",(0,s.jsx)("br",{})]}),(0,s.jsx)(l.zx,{type:"primary",style:{position:"absolute",width:"85%",bottom:10,maxWidth:400},children:"Get Started"})]})}),(0,s.jsx)(l.JX,{md:{span:8},xs:{span:24},style:{display:"flex"},children:(0,s.jsxs)(l.Zb,{title:"Locally hosted (Enterprise)",bordered:!0,style:{width:"100%",marginRight:10},children:["\uD83E\uDDBE Model Type: ",(0,s.jsx)("code",{children:"SQLCoder-34b-instruct"})," ",(0,s.jsx)(l.u,{title:"Our most capable closed-source model with 91% accuracy on `sql-eval`",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83E\uDD1D Annual contracts for on-prem deployment ",(0,s.jsx)(l.u,{title:"The model is hosted on your servers, along with a Docker image for data access, visualization, and other tools",children:"ℹ"})," ",(0,s.jsx)("br",{}),"✅ Follows-instructions ",(0,s.jsx)(l.u,{title:"The model is great at following specialized instructions",children:"ℹ"})," ",(0,s.jsx)("br",{}),"✅ Fine-tuned model ",(0,s.jsx)(l.u,{title:"Model can be fine-tuned great for complex questions that require specialized domain knowledge, like healthcare and finance",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDC77\uD83C\uDFFD‍♂️ Specialized agents (incl healthcare and finance) ",(0,s.jsx)(l.u,{title:"The model is proficient at solving specialist problems involving multiple steps and requiring niche domain knowledge.",children:"ℹ"})," ",(0,s.jsx)("br",{}),"\uD83D\uDD10 Complete data privacy with no data sharing",(0,s.jsx)("hr",{style:{marginTop:"1em",border:"1px solid rgba(0,0,0,0.1)"}}),(0,s.jsxs)("div",{style:{paddingBottom:"2em"},children:[(0,s.jsx)("h3",{children:"Pricing"}),"\uD83D\uDCB0 Pilots at $5k for 8 weeks",(0,s.jsx)("br",{}),"\uD83D\uDCB0 Annual contracts between $60k/yr to $500,000k/yr",(0,s.jsx)("br",{})]}),(0,s.jsx)(l.zx,{type:"primary",ghost:!0,style:{position:"absolute",width:"85%",bottom:10,maxWidth:400},children:"Contact Us"})]})})]})]})]});t.default=o}},function(e){e.O(0,[215,774,888,179],function(){return e(e.s=75557)}),_N_E=e.O()}]);