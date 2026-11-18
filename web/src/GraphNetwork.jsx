import React, {useEffect, useId, useMemo, useRef, useState} from 'react';
import {Network as VisNetwork, DataSet} from 'vis-network/standalone';
import {Maximize, Minus, Plus, Network, FileText, ChevronRight} from 'lucide-react';

export function fromPaths(paths=[]){
  const nodes=new Map(), edges=new Map();
  for(const p of paths){
    const names=p.entities||p.node_chain||[];
    names.forEach((name,i)=>{if(name && !nodes.has(name))nodes.set(name,{name,description:p.descriptions?.[i]||''})});
    (p.edge_types||[]).forEach((label,i)=>{if(!names[i]||!names[i+1])return;const forward=p.directions?.[i]!==false;const source=forward?names[i]:names[i+1],target=forward?names[i+1]:names[i];edges.set(JSON.stringify([source,label,target]),{source,target,label})});
  }
  return {nodes:[...nodes.values()],edges:[...edges.values()]};
}

export function GraphNetwork({data,onSelect,selected,compact=false}){
  const container=useRef(null), network=useRef(null);
  const instanceId=useId();
  useEffect(()=>{
    if(!container.current)return;
    const nodes=new DataSet((data?.nodes||[]).map(n=>({id:n.name,label:n.name,title:n.name,shape:'dot',size:n.name===selected?24:17,color:{background:n.name===selected?'#0f8c7d':'#e5f5f1',border:n.name===selected?'#08796c':'#6bb9a9',highlight:{background:'#0f8c7d',border:'#08796c'}},font:{color:n.name===selected?'#08796c':'#244857',size:13,face:'Microsoft YaHei'},borderWidth:n.name===selected?3:2})));
    const edges=new DataSet((data?.edges||[]).map((e,i)=>({id:`${i}`,from:e.source,to:e.target,label:e.label||'',arrows:{to:{enabled:true,scaleFactor:.55}},color:{color:'#aabbbf',highlight:'#139887'},font:{size:11,color:'#6c8389',strokeWidth:4,strokeColor:'#f8fbfb',align:'middle'},smooth:{type:'dynamic'}})));
    const net=new VisNetwork(container.current,{nodes,edges},{autoResize:true,interaction:{hover:true,navigationButtons:false,keyboard:{enabled:true,bindToWindow:false},multiselect:false},physics:{enabled:true,solver:'forceAtlas2Based',forceAtlas2Based:{gravitationalConstant:-48,centralGravity:.012,springLength:125,springConstant:.06,avoidOverlap:.8},stabilization:{iterations:120}},layout:{improvedLayout:true,randomSeed:14},nodes:{chosen:true},edges:{width:1.2}});
    network.current=net;
    let userMoved=false;
    net.on('dragStart',()=>{userMoved=true});
    net.on('zoom',()=>{userMoved=true});
    net.once('stabilizationIterationsDone',()=>{if(!userMoved){net.fit({animation:false});net.moveTo({scale:net.getScale()*.85,animation:false})}});
    net.on('click',({nodes:ids})=>{if(ids.length)onSelect?.(ids[0])});
    return()=>{network.current=null;net.destroy()};
  },[data,instanceId]);
  useEffect(()=>{if(network.current && selected && data?.nodes?.some(n=>n.name===selected))network.current.selectNodes([selected])},[selected,data]);
  const zoom=factor=>{const n=network.current;if(n)n.moveTo({scale:Math.max(.25,Math.min(3,n.getScale()*factor)),animation:{duration:180}})};
  return <div className={'network-frame '+(compact?'network-compact':'')}>
    {data?.nodes?.length?<><div className="network-canvas" ref={container} aria-label="可交互知识图谱" role="img"/><div className="network-controls"><button onClick={()=>zoom(1.25)} title="放大" aria-label="放大图谱"><Plus size={16}/></button><button onClick={()=>zoom(.8)} title="缩小" aria-label="缩小图谱"><Minus size={16}/></button><button onClick={()=>network.current?.fit({animation:{duration:300}})} title="适应画布" aria-label="适应画布"><Maximize size={15}/></button></div><div className="network-hint">拖拽节点 · 滚轮缩放 · 点击查看详情</div></>:<div className="network-empty"><Network size={23}/><span>暂无可展示的图谱关系</span></div>}
  </div>
}

export function SourceList({sources=[]}){return sources.length?<div className="source-list">{sources.map((s)=><div className="source" key={s.chunk_id}><div className="source-head"><FileText size={16}/><strong>{s.source_doc||'未命名文档'}</strong></div><small>片段 {s.position||'—'} · {s.chunk_id}</small><blockquote>{s.text}</blockquote></div>)}</div>:<p className="muted">暂无原文来源</p>}

export function NodeDetails({node,onExplore,loading=false}){return <div className="node-detail"><div className="detail-heading"><span>节点详情</span>{node&&onExplore&&<button className="textbtn explore-button" onClick={()=>onExplore(node.name)}>探索邻域 <ChevronRight size={15}/></button>}</div>{loading?<p className="muted">正在加载节点信息…</p>:!node?<div className="detail-empty">点击图谱中的节点查看详情</div>:<><div className="detail-name"><span className="detail-symbol"><Network size={17}/></span><div><strong>{node.name}</strong><small>{node.type||'实体'}</small></div></div><div className="detail-description">{node.description||'暂无描述'}</div>{node.aliases?.length>0&&<p className="muted">别名：{node.aliases.join('、')}</p>}<div className="mini-heading">关联来源</div><SourceList sources={node.sources||[]}/></>}</div>}
