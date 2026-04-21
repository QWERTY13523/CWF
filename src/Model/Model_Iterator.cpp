#pragma once
#include "BGAL/Model/Model_Iterator.h"

namespace BGAL
{
  _Face_Iterator::_Face_Iterator(const _Model *in_model, const int &in_cursor)
      : _cursor(in_cursor)
  {
    _model = in_model;
  }
  _Face_Iterator &_Face_Iterator::operator=(const _Face_Iterator &fit)
  {
    _model = fit._model;
    _cursor = fit._cursor;
    return (*this);
  }
  const _Model::_MFace &_Face_Iterator::operator*()
  {
    return _model->_faces[_cursor];
  }
  _Face_Iterator &_Face_Iterator::operator++()
  {
    if (_cursor < _model->number_faces_())
      ++_cursor;
    return (*this);
  }
  _Face_Iterator::~_Face_Iterator()
  {
  }
  _Vertex_Iterator::_Vertex_Iterator(const _Model *in_model, const int &in_cursor)
      : _model(in_model), _cursor(in_cursor)
  {
  }
  _Vertex_Iterator &_Vertex_Iterator::operator=(const _Vertex_Iterator &vit)
  {
    _model = vit._model;
    _cursor = vit._cursor;
    return *this;
  }
  const _Point3 &_Vertex_Iterator::operator*()
  {
    return _model->_vertices[_cursor];
  }
  _Vertex_Iterator &_Vertex_Iterator::operator++()
  {
    if (_cursor < _model->number_vertices_())
      ++_cursor;
    return (*this);
  }
  _Vertex_Iterator::~_Vertex_Iterator()
  {
  }
  _FV_Iterator::_FV_Iterator(const _Model *in_model, const int &in_fid, const int &in_cursor)
      : _model(in_model), _fid(in_fid), _cursor(in_cursor)
  {
  }
  _FV_Iterator &_FV_Iterator::operator=(const _FV_Iterator &fvit)
  {
    _model = fvit._model;
    _fid = fvit._fid;
    _cursor = fvit._cursor;
    return *this;
  }
  const _Point3 &_FV_Iterator::operator*()
  {
    return _model->_vertices[_model->face_(_fid)[_cursor]];
  }
  _FV_Iterator &_FV_Iterator::operator++()
  {
    if (_cursor < 3)
      ++_cursor;
    return (*this);
  }

  _FV_Iterator::~_FV_Iterator()
  {
  }

  _VF_Iterator::_VF_Iterator(const _ManifoldModel *in_model, const int &in_vid, const int &in_cursor)
      : _model(in_model), _vid(in_vid), _cursor(in_cursor)
  {
    _eid = (_cursor < static_cast<int>(_model->_incident_faces_of_vertices[_vid].size()))
               ? _model->_incident_faces_of_vertices[_vid][_cursor]
               : -1;
  }

  _VF_Iterator &_VF_Iterator::operator=(const _VF_Iterator &vfit)
  {
    _model = vfit._model;
    _vid = vfit._vid;
    _cursor = vfit._cursor;
    _eid = vfit._eid;
    return *this;
  }
  const _Model::_MFace &_VF_Iterator::operator*()
  {
    return _model->_faces[id()];
  }
  _VF_Iterator &_VF_Iterator::operator++()
  {
    if (_cursor < static_cast<int>(_model->_incident_faces_of_vertices[_vid].size()))
    {
      ++_cursor;
      _eid = (_cursor < static_cast<int>(_model->_incident_faces_of_vertices[_vid].size()))
                 ? _model->_incident_faces_of_vertices[_vid][_cursor]
                 : -1;
    }
    return (*this);
  }
  _VF_Iterator::~_VF_Iterator()
  {
  }
  _Edge_Iterator::_Edge_Iterator(const _ManifoldModel *in_model, const int &in_cursor)
      : _model(in_model), _cursor(in_cursor)
  {
  }
  _Edge_Iterator &_Edge_Iterator::operator=(const _Edge_Iterator &eit)
  {
    _model = eit._model;
    _cursor = eit._cursor;
    return (*this);
  }
  const _ManifoldModel::_MMEdge &_Edge_Iterator::operator*()
  {
    return _model->_edges[_cursor];
  }
  _Edge_Iterator &_Edge_Iterator::operator++()
  {
    if (_cursor < _model->number_edges_())
    {
      ++_cursor;
    }
    return *this;
  }
  _Edge_Iterator::~_Edge_Iterator()
  {
  }
  _FE_Iterator::_FE_Iterator(const _ManifoldModel *in_model, const int &in_fid, const int &in_cursor)
      : _model(in_model), _fid(in_fid), _cursor(in_cursor)
  {
    _eid = (_cursor < 3) ? _model->_face_edges[_fid][_cursor] : -1;
  }
  _FE_Iterator &_FE_Iterator::operator=(const _FE_Iterator &feit)
  {
    _model = feit._model;
    _fid = feit._fid;
    _cursor = feit._cursor;
    _eid = feit._eid;
    return *this;
  }
  const _ManifoldModel::_MMEdge &_FE_Iterator::operator*()
  {
    return _model->_edges[_eid];
  }
  _FE_Iterator &_FE_Iterator::operator++()
  {
    if (_cursor < 3)
    {
      ++_cursor;
      _eid = (_cursor < 3) ? _model->_face_edges[_fid][_cursor] : -1;
    }
    return *this;
  }
  _FE_Iterator::~_FE_Iterator()
  {
  }
  _FF_Iterator::_FF_Iterator(const _ManifoldModel *in_model, const int &in_fid, const int &in_cursor)
      : _model(in_model), _fid(in_fid), _cursor(in_cursor)
  {
    _eid = -1;
    while (_cursor < 3)
    {
      const int adjacent_face = _model->_face_adjacent_faces[_fid][_cursor];
      if (adjacent_face != -1)
      {
        _eid = adjacent_face;
        break;
      }
      ++_cursor;
    }
  }
  _FF_Iterator &_FF_Iterator::operator=(const _FF_Iterator &ffit)
  {
    _model = ffit._model;
    _fid = ffit._fid;
    _cursor = ffit._cursor;
    _eid = ffit._eid;
    return *this;
  }
  const _Model::_MFace &_FF_Iterator::operator*()
  {
    return _model->_faces[_eid];
  }
  _FF_Iterator &_FF_Iterator::operator++()
  {
    if (_cursor < 3)
    {
      ++_cursor;
    }
    _eid = -1;
    while (_cursor < 3)
    {
      const int adjacent_face = _model->_face_adjacent_faces[_fid][_cursor];
      if (adjacent_face != -1)
      {
        _eid = adjacent_face;
        break;
      }
      ++_cursor;
    }
    return *this;
  }
  _FF_Iterator::~_FF_Iterator()
  {
  }
  _VV_Iterator::_VV_Iterator(const _ManifoldModel *in_model, const int &in_vid, const int &in_cursor)
      : _model(in_model), _vid(in_vid), _cursor(in_cursor)
  {
    _eid = (_cursor < static_cast<int>(_model->_adjacent_vertices_of_vertices[_vid].size()))
               ? _model->_adjacent_vertices_of_vertices[_vid][_cursor]
               : -1;
  }
  _VV_Iterator &_VV_Iterator::operator=(const _VV_Iterator &vvit)
  {
    _model = vvit._model;
    _vid = vvit._vid;
    _cursor = vvit._cursor;
    _eid = vvit._eid;
    return *this;
  }
  const _Point3 &_VV_Iterator::operator*()
  {
    return _model->_vertices[id()];
  }
  _VV_Iterator &_VV_Iterator::operator++()
  {
    if (_cursor < static_cast<int>(_model->_adjacent_vertices_of_vertices[_vid].size()))
    {
      ++_cursor;
      _eid = (_cursor < static_cast<int>(_model->_adjacent_vertices_of_vertices[_vid].size()))
                 ? _model->_adjacent_vertices_of_vertices[_vid][_cursor]
                 : -1;
    }
    return *this;
  }
  _VV_Iterator::~_VV_Iterator()
  {
  }
  _VE_Iterator::_VE_Iterator(const _ManifoldModel *in_model, const int &in_vid, const int &in_cursor)
      : _model(in_model), _vid(in_vid), _cursor(in_cursor)
  {
    _eid = (_cursor < static_cast<int>(_model->_incident_edges_of_vertices[_vid].size()))
               ? _model->_incident_edges_of_vertices[_vid][_cursor]
               : -1;
  }
  _VE_Iterator &_VE_Iterator::operator=(const _VE_Iterator &veit)
  {
    _model = veit._model;
    _vid = veit._vid;
    _cursor = veit._cursor;
    _eid = veit._eid;
    return *this;
  }
  const _ManifoldModel::_MMEdge &_VE_Iterator::operator*()
  {
    return _model->_edges[_eid];
  }
  _VE_Iterator &_VE_Iterator::operator++()
  {
    if (_cursor < static_cast<int>(_model->_incident_edges_of_vertices[_vid].size()))
    {
      ++_cursor;
      _eid = (_cursor < static_cast<int>(_model->_incident_edges_of_vertices[_vid].size()))
                 ? _model->_incident_edges_of_vertices[_vid][_cursor]
                 : -1;
    }
    return *this;
  }
  _VE_Iterator::~_VE_Iterator()
  {
  }
} // namespace BGAL
